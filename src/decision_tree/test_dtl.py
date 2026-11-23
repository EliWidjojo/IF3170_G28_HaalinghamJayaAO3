import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
from src.decision_tree import DecisionTreeCART


def check(condition, description):
    assert condition, description
    print('Lulus:', description)

X = pd.DataFrame({'lead_time': [1,2,3,8,9,10,np.nan,6],
                  'meal': ['BB','BB','HB','HB',None,'HB','BB','BB']})
y = np.array([0,0,0,1,1,1,0,1])
model = DecisionTreeCART(max_depth=4).fit(X, y)
check(np.array_equal(model.predict(X), y), 'klasifikasi fitur campuran')
check(np.allclose(model.predict_proba(X).sum(axis=1),1), 'jumlah probabilitas sama dengan satu')
new = pd.DataFrame({'lead_time':[np.nan,9], 'meal':['UNKNOWN',None]})
check(len(model.predict(new)) == 2, 'prediksi kategori baru dan nilai kosong')
with TemporaryDirectory() as tmp:
    f=Path(tmp)/'cart.pkl'
    model.save(f)
    loaded = DecisionTreeCART.load(f)
    check(np.array_equal(model.predict(X), loaded.predict(X)), 'simpan dan muat model')
Xn = np.array([[1., 2.],[2.,3.],[8.,8.],[9.,9.]])
yn = np.array([0,0,1,1])
check(np.array_equal(DecisionTreeCART(max_depth=2).fit(Xn, yn).predict(Xn), yn), 'fitur numerik NumPy')


# Uji kategori dengan nilai kosong.
Xm = pd.DataFrame({'category': ['A', 'A', None, None]})
ym = np.array([0, 0, 1, 1])
mm = DecisionTreeCART(max_depth=2).fit(Xm, ym)
check(np.array_equal(mm.predict(Xm), ym), 'pemisahan kategori dan nilai kosong')

# Uji kategori baru pada fitur campuran.
Xc = pd.DataFrame({'type': ['a', 'b', 'c', 'c', None, 'a'],
                   'amount': [2.0, 3.0, 7.0, 8.0, 9.0, np.nan]})
yc = np.array([0, 1, 1, 1, 0, 0])
mc = DecisionTreeCART(max_depth=3).fit(Xc, yc)
check(set(mc.predict(pd.DataFrame({'type': ['unknown', None],
                                   'amount': [4.0, np.nan]}))).issubset({0, 1}),
      'prediksi nilai yang belum dikenal')

print('Semua pengujian sintetis lulus')
