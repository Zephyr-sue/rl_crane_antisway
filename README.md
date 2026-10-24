# RL-Based Anti-Sway Control of a Double-Pendulum Crane

A reinforcement learning framework for anti-sway positioning of an overhead double-pendulum crane (trolley, hook, and payload) under large initial sway angles, variable payload masses, changing cable lengths, and external disturbances.

## Project Structure

```
rl_crane_antisway/
├── envs/
│   ├── __init__.py
│   └── double_pendulum_crane_env.py   # Custom Gymnasium environment
├── models/
│   ├── __init__.py                    # RL policy architectures / wrappers
├── utils/
│   ├── __init__.py
│   └── physics.py                     # Nonlinear equations of motion & RK4 integration
├── tests/
│   └── test_physics.py                # Verification tests for dynamics and energy conservation
├── train.py                           # Training entry point (PPO / SAC)
├── evaluate.py                        # Evaluation and benchmark comparisons
├── requirements.txt                   # Dependencies
└── README.md
```

## System Overview
- **Actuator**: Trolley horizontal force $F(t)$
- **Degrees of Freedom**: 
  - $x$: Trolley position
  - $\theta_1$: Hook sway angle
  - $\theta_2$: Payload sway angle
- **Key Features**:
  - Full non-linear Euler-Lagrange equations (handles $> 30^\circ$ sway angles)
  - Variable payload masses ($m_2$) and cable lengths ($L_1, L_2$)
  - Mid-trajectory disturbance injection (pickup shocks, wind gusts)
