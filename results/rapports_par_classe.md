### regles (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      1.000     1.000     1.000        90
      scroll      0.822     0.991     0.898       107
 clic_gauche      0.748     0.902     0.818       132
  clic_droit      0.911     0.543     0.680        94
        drag      0.976     0.776     0.865       107
      neutre      0.771     0.889     0.826        72

    accuracy                          0.852       602
   macro avg      0.871     0.850     0.848       602
weighted avg      0.868     0.852     0.847       602
```

### logreg (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      1.000     1.000     1.000        90
      scroll      0.973     1.000     0.986       107
 clic_gauche      0.963     0.992     0.978       132
  clic_droit      0.970     0.681     0.800        94
        drag      0.800     0.972     0.878       107
      neutre      0.957     0.931     0.944        72

    accuracy                          0.935       602
   macro avg      0.944     0.929     0.931       602
weighted avg      0.942     0.935     0.933       602
```

### logreg+aug (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      1.000     1.000     1.000        90
      scroll      0.981     0.991     0.986       107
 clic_gauche      0.902     0.977     0.938       132
  clic_droit      0.973     0.766     0.857        94
        drag      0.876     0.991     0.930       107
      neutre      0.955     0.875     0.913        72

    accuracy                          0.940       602
   macro avg      0.948     0.933     0.937       602
weighted avg      0.944     0.940     0.939       602
```

### knn (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      0.957     1.000     0.978        90
      scroll      0.982     1.000     0.991       107
 clic_gauche      0.923     1.000     0.960       132
  clic_droit      0.988     0.862     0.920        94
        drag      0.981     0.981     0.981       107
      neutre      0.970     0.903     0.935        72

    accuracy                          0.963       602
   macro avg      0.967     0.958     0.961       602
weighted avg      0.965     0.963     0.963       602
```

### knn+aug (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      0.989     1.000     0.994        90
      scroll      0.982     1.000     0.991       107
 clic_gauche      0.964     1.000     0.981       132
  clic_droit      1.000     0.915     0.956        94
        drag      0.972     0.981     0.977       107
      neutre      0.958     0.944     0.951        72

    accuracy                          0.977       602
   macro avg      0.977     0.973     0.975       602
weighted avg      0.977     0.977     0.977       602
```

### random_forest (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      1.000     1.000     1.000        90
      scroll      0.991     1.000     0.995       107
 clic_gauche      0.936     0.992     0.963       132
  clic_droit      0.979     0.989     0.984        94
        drag      1.000     0.991     0.995       107
      neutre      1.000     0.875     0.933        72

    accuracy                          0.980       602
   macro avg      0.984     0.975     0.979       602
weighted avg      0.981     0.980     0.980       602
```

### random_forest+aug (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      0.989     1.000     0.994        90
      scroll      0.991     1.000     0.995       107
 clic_gauche      0.941     0.970     0.955       132
  clic_droit      0.979     0.989     0.984        94
        drag      1.000     0.991     0.995       107
      neutre      0.939     0.861     0.899        72

    accuracy                          0.973       602
   macro avg      0.973     0.968     0.971       602
weighted avg      0.973     0.973     0.973       602
```

### mlp (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      0.989     1.000     0.994        90
      scroll      0.973     1.000     0.986       107
 clic_gauche      0.874     1.000     0.933       132
  clic_droit      0.987     0.787     0.876        94
        drag      0.945     0.972     0.959       107
      neutre      0.969     0.875     0.920        72

    accuracy                          0.947       602
   macro avg      0.956     0.939     0.945       602
weighted avg      0.950     0.947     0.946       602
```

### mlp+aug (découpage dans le temps)

```
              precision    recall  f1-score   support

 deplacement      0.989     1.000     0.994        90
      scroll      0.991     1.000     0.995       107
 clic_gauche      0.936     1.000     0.967       132
  clic_droit      1.000     0.947     0.973        94
        drag      0.991     0.981     0.986       107
      neutre      0.985     0.917     0.950        72

    accuracy                          0.978       602
   macro avg      0.982     0.974     0.978       602
weighted avg      0.979     0.978     0.978       602
```
