import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
from src.decision_tree import DecisionTreeCART


def check(condition, description):
    assert condition, description
    print('PASS:', description)

X = pd.DataFrame({'lead_time': [1,2,3,8,9,10,np.nan,6],
                  'meal': ['BB','BB','HB','HB',None,'HB','BB','BB']})
y = np.array([0,0,0,1,1,1,0,1])
model = DecisionTreeCART(max_depth=4).fit(X, y)
check(np.array_equal(model.predict(X), y), 'mixed-type classification')
check(np.allclose(model.predict_proba(X).sum(axis=1),1), 'probability rows sum to 1')
new = pd.DataFrame({'lead_time':[np.nan,9], 'meal':['UNKNOWN',None]})
check(len(model.predict(new)) == 2, 'unknown categories and null at inference')
with TemporaryDirectory() as tmp:
    f=Path(tmp)/'cart.pkl'
    model.save(f)
    loaded = DecisionTreeCART.load(f)
    check(np.array_equal(model.predict(X), loaded.predict(X)), 'save/load')
Xn = np.array([[1., 2.],[2.,3.],[8.,8.],[9.,9.]])
yn = np.array([0,0,1,1])
check(np.array_equal(DecisionTreeCART(max_depth=2).fit(Xn, yn).predict(Xn), yn), 'numpy numeric')
print('ALL TESTS PASSED (synthetic only)')
