# coding: utf-8
import os
import shutil
import queue
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time
import pickle
import pandas as pd
import numpy as np
import copy

from torch import Tensor
from batch import Batch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from helpers import load_config, set_seed, load_checkpoint, log_cfg, make_model_dir, make_logger, ConfigurationError, get_latest_checkpoint, symlink_update
from data import load_data, make_data_iter
from model import build_model, Model
from torch.utils.tensorboard import SummaryWriter
from constants import TARGET_PAD
from loss import Loss
from builders import build_gradient_clipper, build_optimizer, build_scheduler
from plot_videos import plot_video, alter_DTW_timing
from prediction import validate_on_data

class TrainManager:
    def __init__(self, model: Model, config: dict, test=False):

        train_config = config["training"]
        model_dir = train_config["model_dir"]

        model_continue = train_config.get("continue", True)

        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        if test:
            self.model_dir = train_config["model_dir"]
        else:
            self.model_dir = make_model_dir(
                model_dir=train_config["model_dir"],
                overwrite=train_config.get("overwrite", False)
            )

        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self.target_pad = TARGET_PAD

        self.loss = Loss(cfg = config, target_pad=self.target_pad)

        self.normalization = "batch"

        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        self.validation_freq = train_config.get("validation_freq", 1000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)

        self.eval_metric = train_config.get("eval_metric", "dtw").lower()
        if self.eval_metric not in ['bleu', 'chrf', "dtw"]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', 'DTW'")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                       "eval_metric")
        
        if self.early_stopping_metric in ["loss","dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', "
                                    "valid options: 'loss', 'dtw',.")
        
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size",self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",self.batch_type)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        self.max_output_length = train_config.get("max_output_length", None)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.to(device)
            self.loss.to(device)

        self.steps = 0
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score
        
        self.skip_frames = config["data"].get("skip_frames", 1)
        
        self._log_parameters_list()
        
        if model_continue:
            ckpt = get_latest_checkpoint(model_dir)
            if ckpt is None:
                self.logger.info("Can't find checkpoint in directory %s", ckpt)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)

    def _log_parameters_list(self) -> None:
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def save_skeleton_files_for_mpje(self, hypotheses, references, inputs, file_paths, data_name, model_dir):
        if isinstance(hypotheses, list):
            hyp_tensors = []
            for hyp in hypotheses:
                if isinstance(hyp, torch.Tensor):
                    hyp_tensors.append(hyp.cpu())
                else:
                    hyp_tensors.append(torch.tensor(hyp).cpu())
            hypotheses_tensor = pad_sequence(hyp_tensors, batch_first=True)
        else:
            hypotheses_tensor = hypotheses.cpu() if hypotheses.is_cuda else hypotheses
        
        if isinstance(references, list):
            ref_tensors = []
            for ref in references:
                if isinstance(ref, torch.Tensor):
                    ref_tensors.append(ref.cpu())
                else:
                    ref_tensors.append(torch.tensor(ref).cpu())
            references_tensor = pad_sequence(ref_tensors, batch_first=True)
        else:
            references_tensor = references.cpu() if references.is_cuda else references

        hyp_path = os.path.join(model_dir, f"{data_name}_hyp_skels.pt")
        torch.save(hypotheses_tensor, hyp_path)
        
        ref_path = os.path.join(model_dir, f"{data_name}_ref_skels.pt")
        torch.save(references_tensor, ref_path)
        
        return hyp_path, ref_path

    def _save_checkpoint(self, type="every") -> None:
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)

        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_best_queue.put(model_path)

            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                symlink_update("{}_best.ckpt".format(self.steps), best_path)
            except OSError:
                torch.save(state, best_path)

        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_queue.put(model_path)

            every_path = "{}/every.ckpt".format(self.model_dir)
            try:
                symlink_update("{}_best.ckpt".format(self.steps), every_path)
            except OSError:
                torch.save(state, every_path)

    def init_from_checkpoint(self, path: str) -> None:
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None and \
                self.scheduler is not None:
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.to(torch.device("cpu"))

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) -> None:
        train_iter = make_data_iter(
            dataset=train_data,
            batch_size=self.batch_size,
            vocab=self.model.src_vocab,
            trg_size=len(self.model.trg_vocab),
            shuffle=True,
            train=True
        )
                
        val_step = 0
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            for batch in iter(train_iter):
                self.model.train()

                batch = Batch(torch_batch=batch,
                              pad_index=self.pad_index,
                              model=self.model)
                
                update = count == 0

                batch_loss = self._train_batch(batch, update=update)

                self.tb_writer.add_scalar("train/train_batch_loss", batch_loss, self.steps)
                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()

                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()

                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths = \
                    validate_on_data(
                        model=self.model,
                        data=valid_data,
                        batch_size=self.eval_batch_size,
                        max_output_length=self.max_output_length,
                        eval_metric=self.eval_metric,
                        loss_function=self.loss,
                        vocab=self.model.src_vocab
                    )
                    
                    self.save_skeleton_files_for_mpje(
                        hypotheses=valid_hypotheses,
                        references=valid_references,
                        inputs=valid_inputs,
                        file_paths=valid_file_paths,
                        data_name="dev",
                        model_dir=self.model_dir
                    )

                    val_step += 1

                    self.tb_writer.add_scalar("valid/valid_loss", valid_loss, self.steps)
                    self.tb_writer.add_scalar("valid/valid_score", valid_score, self.steps)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                        ckpt_score = valid_score
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                        self.produce_validation_video(
                            output_joints=valid_hypotheses,
                            inputs=valid_inputs,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="val_inf",
                            file_paths=valid_file_paths,
                        )

                    self._save_checkpoint(type="every")

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)

                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val",)

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f,  duration: %.4fs',
                            epoch_no+1, self.steps, valid_score,
                            valid_loss, valid_duration)

                if self.stop:
                    break

            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                     self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no+1,
                             epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no+1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)

        self.tb_writer.close()

    def produce_validation_video(self, output_joints, inputs, references, display, model_dir, type, steps="", file_paths=None, dtw_file=None):
        if type != "test":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")
        elif type == "test":
            dir_name = model_dir + "/test_videos/"

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for i in display:
            seq = output_joints[i]
            ref_seq = references[i]
            input = inputs[i]

            gloss_label = input[0]
            if input[1] != "</s>":
                gloss_label += "_" + input[1]
            if input[2] != "</s>":
                gloss_label += "_" + input[2]

            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)
            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))

            if file_paths is not None:
                sequence_ID = file_paths[i]
            else:
                sequence_ID = None

            print(sequence_ID + '    dtw: ' + '{0:.2f}'.format(float(dtw_score)))

            if dtw_file != None:
                dtw_file.writelines(sequence_ID + ' ' + '{0:.2f}'.format(float(dtw_score)) + '\n')

            plot_video(joints=timing_hyp_seq,
                       file_path=dir_name,
                       video_name=video_ext,
                       references=ref_seq_count,
                       skip_frames=self.skip_frames,
                       sequence_ID=sequence_ID)

    def save_skels(self, output_joints, display, model_dir, type, file_paths=None):
        picklefile = open(model_dir + "/phoenix14t.skels.%s" % type, "wb")

        try:
            csvIn = pd.read_csv(model_dir + "/csv/%s_phoenix2014t.csv" % type, sep='|',encoding='utf-8')
        except FileNotFoundError:
            print(f"Warning: CSV file not found at {model_dir}/csv/{type}_phoenix2014t.csv")
            print("Saving skeleton data without CSV metadata...")
            pickle_list = []
            for i in display:
                name = file_paths[i] if file_paths else f"sequence_{i}"
                seq = output_joints[i].cpu()[:,:-1]
                sign = seq.clone().detach().to(torch.float32)
                dict_num = {'name': name, 'signer': 'unknown', 'gloss': 'unknown', 'text': 'unknown', 'sign': sign}
                pickle_list.append(dict_num)
            pickle.dump(pickle_list, picklefile)
            print("The skeletons of %s data have been saved without metadata." % type)
            return

        pickle_list = []

        for i in display:
            name = file_paths[i]
            video = os.path.basename(name)
            matching_rows = csvIn[csvIn['id'] == video]

            if matching_rows.empty:
                print(f"Warning: No exact match for video id={video}")
                partial_matches = csvIn[csvIn['id'].str.contains(video.split('_')[0], na=False)]
                if not partial_matches.empty:
                    signer = partial_matches.iloc[0]['signer']
                    gloss = partial_matches.iloc[0]['annotation'] 
                    text = partial_matches.iloc[0]['translation']
                else:
                    signer = 'unknown'
                    gloss = 'unknown' 
                    text = 'unknown'
            else:
                signer = matching_rows.iloc[0]['signer']
                gloss = matching_rows.iloc[0]['annotation']
                text = matching_rows.iloc[0]['translation']

            seq = output_joints[i].cpu()[:,:-1]
            sign = seq.clone().detach().to(torch.float32)

            dict_num = {'name': name, 'signer': signer, 'gloss': gloss, 'text': text, 'sign': sign}
            pickle_list.append(dict_num)

        pickle.dump(pickle_list, picklefile)
        print("The skeletons of %s data have been saved." % type)

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:
        batch_loss = self.model.get_loss_for_batch(
            is_train=True,
            batch=batch,
            loss_function=self.loss
        )

        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss = batch_loss / normalizer
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1

        self.total_tokens += batch.ntokens
        return norm_batch_loss

    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str, new_best: bool = False, report_type: str = "val") -> None:
        current_lr = -1
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))


def train(cfg_file: str, ckpt=None):
    cfg = load_config(cfg_file)
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

    model = build_model(cfg=cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    if model is None:
        raise ValueError("Model creation failed. Check your build_model function.")

    if ckpt is not None:
        use_cuda = cfg["training"].get("use_cuda", True)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
        checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
        if checkpoint is None or "model_state" not in checkpoint:
            raise ValueError(f"Failed to load model state from checkpoint: {ckpt}")
        state_dict = checkpoint["model_state"]
        if "src_embed.lut.weight" in state_dict:
            state_dict.pop("src_embed.lut.weight")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    trainer = TrainManager(config=cfg, model=model, test=False)

    shutil.copy2(cfg_file, os.path.join(trainer.model_dir, "Sign-IDD.yaml"))
    log_cfg(cfg, trainer.logger)

    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)


def test(cfg_file: str, ckpt: str = None):
    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in directory: {model_dir}")

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

    model = build_model(cfg=cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    if model is None:
        raise ValueError("Model creation failed. Check your build_model function.")

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
    checkpoint = load_checkpoint(ckpt, use_cuda=cfg["training"].get("use_cuda", True))
    if checkpoint is None or "model_state" not in checkpoint:
        raise ValueError(f"Failed to load model state from checkpoint: {ckpt}")

    state_dict = checkpoint["model_state"]
    emb_key = "src_embed.lut.weight"

    model_emb = None
    if hasattr(model, "src_embed"):
        src_embed_obj = getattr(model, "src_embed")
        if hasattr(src_embed_obj, "lut") and hasattr(src_embed_obj.lut, "weight"):
            model_emb = src_embed_obj.lut.weight
        elif hasattr(src_embed_obj, "weight"):
            model_emb = src_embed_obj.weight

    if emb_key in state_dict:
        ckpt_shape = state_dict[emb_key].shape
        model_shape = model_emb.shape if model_emb is not None else None
        if model_shape is None or ckpt_shape != model_shape:
            print(f"[INFO] Skipping {emb_key} due to shape mismatch: checkpoint {ckpt_shape} vs model {model_shape}")
            state_dict.pop(emb_key)
        else:
            pass

    model.load_state_dict(state_dict, strict=False)

    use_cuda_flag = cfg["training"].get("use_cuda", True)
    device_local = torch.device("cuda" if use_cuda_flag and torch.cuda.is_available() else "cpu")
    model.to(device_local)

    cfg_no_continue = copy.deepcopy(cfg)
    cfg_no_continue["training"] = dict(cfg_no_continue.get("training", {}))
    cfg_no_continue["training"]["continue"] = False

    trainer = TrainManager(model=model, config=cfg_no_continue, test=True)

    data_to_predict = {"dev": dev_data, "test": test_data}
    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    for data_name, dataset in data_to_predict.items():
        score, loss, references, hypotheses, inputs, all_dtw_scores, file_paths = validate_on_data(
            model=model,
            data=dataset,
            batch_size=batch_size,
            max_output_length=max_output_length,
            eval_metric=eval_metric,
            loss_function=None,
            vocab=model.src_vocab
        )
        
        trainer.save_skeleton_files_for_mpje(
            hypotheses=hypotheses,
            references=references,
            inputs=inputs,
            file_paths=file_paths,
            data_name=data_name,
            model_dir=model_dir
        )
        
        output_dir = os.path.join(model_dir, "test_videos")
        os.makedirs(output_dir, exist_ok=True)

        dtw_path = os.path.join(output_dir, f"{data_name}_dtw.txt")
        with open(dtw_path, "w") as f:
            f.write(f"DTW Score of {data_name} set: {score:.3f}\n")
        print(f"DTW Score of {data_name} set: {score:.3f}")

        display = list(range(len(hypotheses)))

        trainer.save_skels(output_joints=hypotheses, display=display, model_dir=model_dir, type=data_name, file_paths=file_paths)
        with open(dtw_path, "a") as dtw_file:
            trainer.produce_validation_video(
                output_joints=hypotheses,
                inputs=inputs,
                references=references,
                model_dir=model_dir,
                display=display,
                type="test",
                file_paths=file_paths,
                dtw_file=dtw_file,
            )
    
    return model_dir