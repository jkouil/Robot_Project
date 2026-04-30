# Pupper-like model notes

This is not an official Pupper model. I made a simplified MuJoCo model with a similar size and 12-DoF leg layout, using the public Stanford Pupper / StanfordQuadruped pages and MangDang pages as references.

The numbers are approximate. I used simple boxes, capsules, and spheres instead of CAD meshes.

Main dimensions used:

- torso: about 0.24 m long, 0.11 m wide, 0.05 m tall
- torso mass: 1.2 kg
- target total mass: about 2.2 kg
- hip x offset: 0.095 m
- hip y offset: 0.055 m
- upper leg length: 0.10 m
- lower leg length: 0.11 m
- foot radius: 0.018 m

Joint order:

```text
fl_abduction, fl_hip, fl_knee,
fr_abduction, fr_hip, fr_knee,
rl_abduction, rl_hip, rl_knee,
rr_abduction, rr_hip, rr_knee
```

Default stand pose:

```text
left abduction  = +0.05 rad
right abduction = -0.05 rad
hips            =  0.75 rad
knees           = -1.35 rad
```

The model also has:

- an `imu_site` on the torso
- a `front_camera` on the front of the torso
- position actuators for the 12 joints

The XML files are:

- `models/pupper_like.xml`
- `models/pupper_like_preview_terrain.xml`

The second file is the one used for the terrain experiments.
