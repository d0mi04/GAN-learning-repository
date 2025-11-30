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

### 3. Loss range in DCGAN should be:
## 📌 What is a healthy loss range in DCGAN?

| Metric | Healthy behavior | What it means |
|--------|------------------|---------------|
| lossD | 0.5 – 1.5 | balance between "sometimes G wins, sometimes D" |
| lossG | 0.5 – 2.0 | generator receives sensible gradients |
| loss oscillations | YES | this is normal in GANs |
| stable lossD ≈ 2.0 | BAD | D too strong |
| lossG < 0 | BAD | gradient separation / saturation |

In `GAN-first-approach` we can observe:
- Discriminator's domination: 
```
lossD ~ 1.9–2.0
```
- generator's loss:
```
lossG ~ 0 → -0.4
```
That means the model most likely is NOT generating correct digits. Too strong D kills G --> the generator stops improving.

### 4. How to visually check if DCGAN is working
Ideal behavior curve:
- at the begining:
```
lossD grows ~1.0 → 1.5
lossG drops ~3.0 → 1.0
```
- then:
```
lossG oscillates: 0.8–2.0
lossD oscillates: 0.7–1.5
```
- images become increasingly sharp.

### 5. Fixing occurred problems
#### Poblem 1: Discriminator is too strong
Solutions:
- **weaker D architecture**
    - reduce number of filters, e.g. from 64 to 32.
- **label smoothing in D** to gently stabilize training:
```python
real_labels = torch.ones(batch_size) * 0.9
```
- **add Gaussian noise on D input**, add slight noise to real/fake images
```python
x = x + 0.05 * torch.randn_like(x)
```
#### Problem 2: `z_dim` too low or bad initialization
- set `z_dim = 100` or `dim_z = 128`, MNIST is a simple dataset, but 20-50 would be too low
- check if weight initialization was correct
```python 
def weights_init(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```
```python
model.apply(weights_init)
```
#### Problem 3: Lack of iteration balancing
DCGAN works better when doing this:
```
👉 1 krok D
👉 1 krok G
```
NOT more Discriminator steps per one generator step.

## Propozycja zmian:
