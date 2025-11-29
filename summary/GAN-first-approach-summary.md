# Summary
in the logs we can observe that this behavior is not good. 
```
lossD = 1.89–2.01
lossG = 0.3, 0.1, -0.2, -0.4
```

### 1. the Discriminator is extremely confident `lossD ~ 2.0` this means
 - almost all fake images are classified as `FAKE`
 - almost all real images are classified as `REAL`
 - the Generator is completely failing.

Discriminator loss close to 2.0 means (for Binary Cross Entropy, BCE): 
- 👉 `log(D(real)) ≈ 0`
- 👉 `log(1 − D(fake)) ≈ 0`

and that means: 👉 `D(real) → 1, D(fake) → 0` the Discriminator become too strong that it has dominated the training.

### 2. Generator loss is often negative 
In classic DCGAN, generator loss should NOT be negative. This is a signal that:
- the generator is "giving up"
- gradients become very small/unstable
- the discriminator is too confident --> the generator receives too little signal for learning.
