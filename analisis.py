"""Funciones del Proyecto 1. No dependen de la interfaz de Streamlit."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

TARGET = 'Diabetes_012'
FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
BINARY = FEATURES[:3] + ['Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex']
RANGES = {'BMI': (1,100), 'GenHlth': (1,5), 'MentHlth': (0,30), 'PhysHlth': (0,30), 'Age': (1,13), 'Education': (1,6), 'Income': (1,8)}
LABELS = ['Sin diabetes / solo gestacional', 'Prediabetes', 'Diabetes']

def preparar(raw):
    """Valida esquema y códigos; no elimina perfiles repetidos ni imputa aquí."""
    missing = [c for c in [TARGET]+FEATURES if c not in raw]
    if missing:
        raise ValueError('Faltan columnas: '+', '.join(missing))
    df = raw[[TARGET]+FEATURES].copy()
    registro = []
    for c in df:
        original = df[c]
        values = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & values.isna()
        if c == TARGET:
            invalid |= values.notna() & ~values.isin([0,1,2])
        elif c in BINARY:
            invalid |= values.notna() & ~values.isin([0,1])
        else:
            lo, hi = RANGES[c]
            invalid |= values.notna() & (~values.between(lo,hi) | (values % 1 != 0))
        registro.append({'Variable':c,'Nulos originales':int(original.isna().sum()),'Valores inválidos':int(invalid.sum())})
        df[c] = values.mask(invalid)
    excluded = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    if df.empty or set(df[TARGET].unique()) != {0,1,2}:
        raise ValueError('Se necesitan registros válidos de las tres categorías 0, 1 y 2.')
    return df, pd.DataFrame(registro), excluded

def eda(df):
    rows=[]
    for i,c in enumerate(df):
        s=df[c]
        categorical=c == TARGET or c in BINARY or c in ['GenHlth','Age','Education','Income']
        row={'Variable':c,'Posición':i+1,'Tipo conceptual':'Categórica codificada' if categorical else 'Numérica','Tipo pandas':str(s.dtype),'Filas':len(s),'No nulos':int(s.count()),'Nulos':int(s.isna().sum()),'Valores distintos':s.nunique(),'Niveles (máx. 5)':', '.join(map(str,sorted(s.dropna().unique()))) if s.nunique()<=5 else ''}
        for name,value in [('Mínimo',s.min()),('Máximo',s.max()),('Promedio',s.mean()),('DS',s.std()),('Mediana',s.median()),('Q1',s.quantile(.25)),('Q3',s.quantile(.75))]:
            row[name]=np.nan if categorical else value
        rows.append(row)
    return pd.DataFrame(rows)

def metricas(y,p):
    return {'Exactitud':accuracy_score(y,p),'Exactitud balanceada':balanced_accuracy_score(y,p),'F1 macro':f1_score(y,p,average='macro',zero_division=0)}

def entrenar(df, trees=100, importance_n=3000):
    """Perfiles iguales se mantienen en un mismo subconjunto para evitar fuga."""
    X,y=df[FEATURES],df[TARGET]
    groups=pd.util.hash_pandas_object(X,index=False)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    dev,test=next(splitter.split(X,y,groups))
    Xdev,ydev=X.iloc[dev],y.iloc[dev]
    inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=43)
    train,val=next(inner.split(Xdev,ydev,groups.iloc[dev]))
    Xtrain,ytrain=Xdev.iloc[train],ydev.iloc[train]
    Xval,yval=Xdev.iloc[val],ydev.iloc[val]
    Xtest,ytest=X.iloc[test],y.iloc[test]
    if any(set(v.unique()) != {0,1,2} for v in [ytrain,yval,ytest]):
        raise ValueError('La separación no conserva las tres clases. Usa el dataset completo.')
    if any(Xtrain[c].notna().sum()==0 for c in FEATURES):
        raise ValueError('Hay una variable sin datos válidos en entrenamiento.')
    models={
        'Base: clase mayoritaria':Pipeline([('imputar',SimpleImputer(strategy='median')),('modelo',DummyClassifier(strategy='most_frequent'))]),
        'Regresión logística':Pipeline([('imputar',SimpleImputer(strategy='median')),('escalar',StandardScaler()),('modelo',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))]),
        'Bosque aleatorio':Pipeline([('imputar',SimpleImputer(strategy='median')),('modelo',RandomForestClassifier(n_estimators=trees,max_depth=12,min_samples_leaf=10,class_weight='balanced_subsample',random_state=42,n_jobs=-1))])}
    validation=[]
    for name,model in models.items():
        model.fit(Xtrain,ytrain)
        validation.append({'Modelo':name,**metricas(yval,model.predict(Xval))})
    table=pd.DataFrame(validation).set_index('Modelo')
    # La elección ocurre en validación; la prueba NO elige el modelo.
    winner=table.drop(index='Base: clase mayoritaria')['F1 macro'].idxmax()
    selected=models[winner]
    pred=selected.predict(Xtest)
    report=pd.DataFrame(classification_report(ytest,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)).T
    # Muestra aleatoria reproducible SOLO de validación para reducir tiempo.
    rng=np.random.default_rng(42)
    positions=rng.choice(len(Xval),size=min(importance_n,len(Xval)),replace=False)
    imp=permutation_importance(selected,Xval.iloc[positions],yval.iloc[positions],scoring='f1_macro',n_repeats=3,random_state=42,n_jobs=1)
    importance=pd.DataFrame({'Variable':FEATURES,'Caída media de F1 macro':imp.importances_mean,'DS entre repeticiones':imp.importances_std}).sort_values('Caída media de F1 macro',ascending=False)
    splits=pd.DataFrame({name:values.value_counts().reindex([0,1,2],fill_value=0) for name,values in [('Entrenamiento',ytrain),('Validación',yval),('Prueba',ytest)]})
    assert set(groups.iloc[dev].iloc[train]).isdisjoint(set(groups.iloc[test]))
    assert set(groups.iloc[dev].iloc[train]).isdisjoint(set(groups.iloc[dev].iloc[val]))
    return {'validacion':table,'ganador':winner,'prueba':metricas(ytest,pred),'reporte':report,'matriz':pd.DataFrame(confusion_matrix(ytest,pred,labels=[0,1,2]),index=LABELS,columns=LABELS),'importancia':importance,'particiones':splits,'modelo':selected}
