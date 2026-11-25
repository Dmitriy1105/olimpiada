"""Experimental teacher for the published toy lander simulator (CELL 10).
Consumes only the 42 normalized observation features; returns actions 0..8.
Not a real-world flight controller or a submission checkpoint.
"""
import torch

@torch.no_grad()
def teacher_action(obs):
    dx, y, vx, vy = obs[:, 0]*20, obs[:, 1]*40, obs[:, 2]*10, obs[:, 3]*10
    theta = torch.atan2(obs[:, 4], obs[:, 5])
    omega = obs[:, 6]*3
    g, thrust = obs[:, 8]*10, obs[:, 9]*20
    left, right = obs[:, 10]*5, obs[:, 11]*5
    wind, drag, angular_drag = obs[:, 12]*3, obs[:, 13]*.15, obs[:, 14]
    width = obs[:, 15]*4
    contact = obs[:, 39:41].amax(dim=1) > .5

    vx_ref = (-.8*dx).clamp(-4.5, 4.5)
    ax = 1.65*(vx_ref-vx)
    theta_ref = torch.atan2(wind-drag*vx-ax, g)
    authority = torch.acos((1.12*g/thrust).clamp(0,.999))
    tilt_limit = torch.minimum(authority, (.13+.11*y.clamp_min(0)).clamp(max=.60))
    theta_ref = torch.maximum(-tilt_limit, torch.minimum(tilt_limit, theta_ref))

    aligned = (dx.abs() < (width-.8).clamp_min(.35)) & (vx.abs() < .65)
    clearance = torch.where(aligned, torch.zeros_like(y), 10+dx.abs().clamp(max=7))
    vy_ref = -(0.45+0.55*(y-clearance).clamp_min(0).sqrt()).clamp(max=4.5)
    vy_ref = torch.where(~aligned & (y<clearance), (.9*(clearance-y)).clamp(max=2.5), vy_ref)
    thrust_ref = (g+drag*vy+2.5*(vy_ref-vy))/(thrust*theta.cos().clamp(min=.15))
    throttle = (2*thrust_ref).round().clamp(0,2).long()

    acc = 7.0*(theta_ref-theta)-(3.5-angular_drag)*omega
    turn = torch.where(acc > .28*right, 1, torch.where(acc < -.28*left,-1,0))
    settled_zone = contact & (dx.abs() < (width-.75).clamp_min(.3)) & (vx.abs()<.6) & (theta.abs()<.20)
    throttle = torch.where(settled_zone,0,throttle)
    turn = torch.where(settled_zone,0,turn)
    return (3*throttle+turn+1).long()
