# How To Run

## Load Image Stack

for the hw input, uncomment 
```
I = readimage("../data/input_{:d}.tif".format(i+1))
```

change I to 7

```
I, rows, cols = loadIstack()
```

## Uncalibrated/Calibrated Stereo

```
B = uncali(I, rows, cols)
B = cali(I, rows, cols)
```

## Enforce Integrability

```
B = enforce(B)
```

## Integration 

```
Z = integrate(B)
```

## Display Albedo and Normal Based on matrix B

```
displayb(B)
```

## Display the 3D surface based on Z matrix

```
displayz(Z)
```