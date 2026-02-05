# SignPINN
SIGNPINN: PINN INSPIRED DUAL-RATE DIFFUSION FOR SIGN LANGUAGE PRODUCTION
Diffusion models have recently become an effective way
to create sign language poses. This generation in current
models have difficulty retaining the intricate details of hand
movements. Although they improved robustness, they gen-
erally treat all joints uniformly. This overlooks the distinct
dynamics and noise sensitivity of hand motion. This work
proposes SignPINN, a framework that is based on diffu-
sion. It mainly addresses these challenges through two key
contributions. Firstly, we introduce a dual-rate diffusion
process that assigns separate noise schedules to hand and
body joints. This preserves high-frequency hand articula-
tion while maintaining stable global motion using a shared
denoiser. Secondly, we incorporate physics-informed Reg-
ularization as training constraints. This enforces smooth
motion, orientation consistency, and kinematic integrity. Ex-
periments on Phoenix-2014T and How2Sign demonstrate
consistent improvements in evaluation metrics. Results
show the effectiveness of combining part-aware diffusion
dynamics with physics-informed constraints for generat-
ing accurate and physically plausible sign language motion.
