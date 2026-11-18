"""Panel para analistas clínicos. Ejecutar desde esta carpeta: python -m streamlit run app.py"""
from pathlib import Path
from io import BytesIO
import hashlib
import altair as alt
import pandas as pd
import streamlit as st
from analisis import TARGET, FEATURES, BINARY, RANGES, LABELS, preparar, eda, entrenar
from contenido import CLASES, LEYENDA, COLORES, VARIABLES, FUENTES, nombre, significado

st.set_page_config(page_title='Diabetes | Análisis clínico',page_icon='📊',layout='wide')
st.markdown('''<style>
.stApp {background-color:#FFFFFF;color:#172033;}
[data-testid="stSidebar"] {background-color:#F1F5F9;}
h1,h2,h3 {color:#173B70;}
[data-testid="stMetric"] {background:#F1F5F9;border-top:4px solid #1D4ED8;padding:16px;border-radius:10px;}
[data-testid="stMetric"]:nth-child(2) {border-color:#15803D;}
.hero {border-left:5px solid #7E22CE;background:#F5F3FF;padding:18px 22px;border-radius:10px;margin-bottom:18px;}
.legend {background:#EFF6FF;border:1px solid #BFDBFE;padding:12px 16px;border-radius:8px;color:#173B70;line-height:1.65;}
/* Selectores semánticos para Streamlit actual y versiones anteriores. */
[role="tablist"] {
    gap:10px !important; padding:8px 3px 16px !important;
    border-bottom:none !important; background:transparent !important;
}
[role="tablist"] [role="tab"] {
    appearance:none !important;
    background:linear-gradient(115deg,#DBEAFE 0%,#E9D5FF 100%) !important;
    color:#243B67 !important; border:1px solid #A5B4FC !important;
    border-radius:12px !important; padding:13px 19px !important;
    min-height:52px !important; height:auto !important;
    box-shadow:0 2px 6px rgba(29,78,216,.12) !important;
    text-decoration:none !important;
    transition:box-shadow .18s ease,background .18s ease;
}
[role="tablist"] [role="tab"] p {
    color:inherit !important; font-weight:600 !important; font-size:15px !important;
}
[role="tablist"] [role="tab"][aria-selected="true"],
[role="tablist"] [role="tab"][data-selected="true"] {
    background:linear-gradient(115deg,#1D4ED8 0%,#7E22CE 100%) !important;
    color:#FFFFFF !important; border-color:#4F46E5 !important;
    box-shadow:0 4px 12px rgba(79,70,229,.25) !important;
}
[role="tablist"] [role="tab"]:hover {
    box-shadow:0 4px 12px rgba(79,70,229,.24) !important;
}
[role="tablist"] [role="tab"]:focus-visible {
    outline:3px solid #F59E0B !important; outline-offset:3px !important;
}
[role="tablist"]::after,
[role="tablist"] .react-aria-SelectionIndicator,
[role="tablist"] [data-baseweb="tab-highlight"],
[role="tablist"] [data-baseweb="tab-border"] {display:none !important;}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    padding:8px 0; border-bottom:1px solid #E2E8F0;
}
[data-testid="stForm"] [data-testid="stHorizontalBlock"]:last-of-type {border-bottom:0;}
</style>''',unsafe_allow_html=True)
st.title('Indicadores de salud y estado de diabetes')
st.caption('Leonela Martínez Loría · Proyecto 1')
st.markdown('<div class="hero"><b>Para analistas clínicos</b><br>Explora datos de encuesta, compara clasificadores y reconoce las variables que aportan información al modelo.</div>',unsafe_allow_html=True)

def leyenda():
    st.markdown(f'<div class="legend">{LEYENDA}</div>',unsafe_allow_html=True)

leyenda()
st.sidebar.header('Panel de análisis clínico')
st.sidebar.markdown('**BRFSS**: Behavioral Risk Factor Surveillance System, un sistema de encuestas telefónicas de salud de Estados Unidos.')
st.sidebar.markdown('**Clases de diabetes**\n\n0: sin diabetes / solo embarazo\n\n1: prediabetes\n\n2: diabetes')
uploaded=st.sidebar.file_uploader('Cargar CSV de Kaggle',type=['csv'])

@st.cache_data
def cargar(data):
    return pd.read_csv(BytesIO(data))

@st.cache_resource(show_spinner=False)
def calcular(df):
    # Se conserva la configuración original del bosque balanceado de la versión 1.1.
    return entrenar(df)


def barras(datos, eje, titulo, colores=None):
    """Barras verticales con etiquetas del eje X siempre horizontales."""
    chart=alt.Chart(datos).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
        x=alt.X(f'{eje}:N',sort=list(datos[eje]),title=titulo,axis=alt.Axis(labelAngle=0,labelOverlap=False)),
        y=alt.Y('Registros:Q',title='Número de respuestas'),
        tooltip=list(datos.columns))
    if colores:
        chart=chart.encode(color=alt.Color(f'{eje}:N',scale=alt.Scale(domain=list(datos[eje]),range=colores),legend=None))
    else:
        chart=chart.mark_bar(color='#7E22CE',cornerRadiusTopLeft=4,cornerRadiusTopRight=4)
    st.altair_chart(chart.properties(height=300).configure_view(stroke=None),theme=None,width='stretch')

local=Path(__file__).parent/'diabetes_012_health_indicators_BRFSS2015.csv'
intro,explore,models,simulation,dictionary=st.tabs(['Proyecto y contexto','Exploración y limpieza','Predicción y resultados','Simular un perfil','Diccionario y términos'])
with intro:
    st.subheader('Público meta y propósito')
    st.write('Público meta: analistas clínicos que exploran indicadores de salud y evalúan modelos de clasificación para investigación. La herramienta permite revisar calidad, distribución de respuestas, desempeño por categoría e importancia de variables.')
    st.write('Objetivo: evaluar si 21 indicadores de salud, hábitos y características demográficas permiten clasificar el estado reportado de diabetes en adultos de esta encuesta.')
    st.markdown('**Pregunta central:** ¿En qué medida un modelo puede clasificar el estado de diabetes a partir de los indicadores disponibles?\n\n**Preguntas específicas:**\n1. ¿Con qué desempeño las variables BRFSS permiten predecir la categoría de diabetes?\n2. ¿Qué variables aportan mayor información al modelo para predecir esa categoría?')
    st.subheader('¿Qué significa BRFSS?')
    st.write('Behavioral Risk Factor Surveillance System: Sistema de Vigilancia de Factores de Riesgo del Comportamiento. Es un sistema de encuestas telefónicas de Estados Unidos sobre conductas de salud, enfermedades crónicas y acceso a servicios. Participan adultos de 18 años o más. Aquí utilizamos una versión seleccionada y recodificada de BRFSS 2015, publicada por Alex Teboul en Kaggle.')
    st.subheader('¿Es un conjunto de datos solo de mujeres?')
    st.write('No. Incluye mujeres y hombres. En este CSV, Sex: 0 = mujer y 1 = hombre. El significado de 0 y 1 depende de la columna: no se debe confundir Sex con Diabetes_012.')
    st.subheader('¿Por qué aparece diabetes durante el embarazo?')
    st.write('No es una variable predictora independiente. Es una respuesta del cuestionario original que, en la versión derivada usada aquí, queda agrupada dentro de la categoría 0 junto con quienes reportaron no tener diabetes. La mención aclara cómo se construyó el resultado; no indica que estemos estudiando solo embarazadas.')
    st.warning('El CSV ya no permite separar a quienes reportaron diabetes solo durante el embarazo del resto de la categoría 0. No podemos contar esos casos ni estimar su importancia como predictor con este archivo. Esa agrupación tampoco significa que ambos grupos sean clínicamente equivalentes.')
    st.subheader('¿Qué se quiere lograr con la predicción?')
    st.write('Dadas las 21 respuestas de un perfil, asignar la categoría que estima el modelo y estudiar su desempeño en registros reservados. No basta contar cuántos factores tiene una persona: cada variable y sus combinaciones pueden aportar información diferente.')
    st.info('Este archivo es transversal: no sigue a las personas hacia el futuro. Por eso la pregunta evaluable es «¿qué categoría reportada predice el modelo?», no «¿voy a desarrollar diabetes?». Para pronosticar aparición futura se necesitarían datos longitudinales y un horizonte temporal definido.')
    st.subheader('Fuentes y documentación')
    for label,url in FUENTES.items(): st.markdown(f'- [{label}]({url})')

with dictionary:
    st.subheader('Diccionario de las 22 columnas')
    st.caption('21 predictores y una variable de resultado. Definiciones del CSV derivado; algunos códigos difieren del cuestionario original CDC.')
    dictionary_df=pd.DataFrame([{'Variable':c,'Nombre en español':v[0],'Qué representa':v[1],'Códigos / unidades':v[2]} for c,v in VARIABLES.items()])
    st.dataframe(dictionary_df,hide_index=True,width='stretch')
    st.download_button('Descargar diccionario',dictionary_df.to_csv(index=False).encode('utf-8-sig'),'diccionario_variables.csv','text/csv')
    st.subheader('Términos del panel')
    for term,definition in {
        'Registro':'Una fila con las respuestas de una persona; no un diagnóstico confirmado por laboratorio.',
        'Predictor':'Una característica usada como entrada. Diabetes_012 es la salida y no entra como predictor.',
        'EDA':'Análisis exploratorio: revisar distribuciones, tipos, nulos y estadísticas antes de modelar.',
        'Nulo':'Celda sin información o valor que se marcó como faltante al validar.',
        'Imputación':'Completar predictores faltantes con medianas aprendidas solo en entrenamiento.',
        'Clase desbalanceada':'Categoría que tiene muchos más o muchos menos registros que las otras.',
        'Entrenamiento / validación / prueba':'Aprender / comparar y seleccionar / evaluar el modelo elegido.',
        'Exactitud':'Proporción total de aciertos. Una clase muy frecuente puede dominar esta cifra.',
        'Precisión (precision)':'De los casos predichos como una clase, proporción que realmente pertenece a esa clase.',
        'Recall o sensibilidad por clase':'De los casos reales de una clase, proporción que el modelo detecta.',
        'F1':'Media armónica de precisión y recall por clase. Va de 0 a 1; un valor mayor es mejor.',
        'F1 macro':'Promedio simple de los tres F1 por clase. Cada categoría pesa lo mismo; no es el porcentaje total de aciertos.',
        'Exactitud balanceada':'Promedio de los recall de las tres clases.',
        'Importancia por permutación':'F1 original menos F1 al desordenar una variable. Positivo: el F1 bajó; negativo: subió; cerca de cero: cambió poco. No demuestra causalidad.',
        'DS / Q1 / Q3':'Desviación estándar / primer cuartil (25 %) / tercer cuartil (75 %).',
    }.items():
        st.markdown(f'**{term}:** {definition}')

if uploaded is None and not local.exists():
    with explore: st.info('Carga el CSV en la barra lateral para revisar datos y limpieza.')
    with models: st.info('Carga el CSV para entrenar y evaluar.')
    with simulation: st.info('Carga el CSV y entrena los modelos para habilitar la simulación.')
    st.stop()
try:
    data=uploaded.getvalue() if uploaded else local.read_bytes()
    signature=hashlib.sha256(data).hexdigest()
    raw=cargar(data)
    df,quality,excluded=preparar(raw)
except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as exc:
    st.error(f'No se pudo procesar el archivo: {exc}')
    st.stop()
if st.session_state.get('firma')!=signature:
    st.session_state.pop('resultado',None)
    st.session_state.pop('perfil_predicho',None)
    st.session_state['firma']=signature

with explore:
    c1,c2,c3=st.columns(3)
    c1.metric('Registros originales',f'{len(raw):,}')
    c2.metric('Predictores',len(FEATURES))
    c3.metric('Celdas nulas originales',int(raw.isna().sum().sum()))
    st.subheader('Vista de los datos originales')
    st.caption('Primeras 100 filas. El análisis usa todos los registros válidos. Los códigos se interpretan según la columna; consulta el diccionario.')
    leyenda()
    st.dataframe(raw.head(100),width='stretch')
    st.subheader('Distribución del estado reportado de diabetes')
    counts=df[TARGET].value_counts().reindex([0,1,2],fill_value=0)
    distribution=pd.DataFrame({'Código':[str(c) for c in CLASES],'Categoría':list(CLASES.values()),'Registros':counts.to_numpy(),'Porcentaje':counts.to_numpy()/len(df)*100})
    barras(distribution,'Código','Categoría de diabetes',COLORES)
    leyenda()
    st.dataframe(distribution.drop(columns='Código'),hide_index=True,width='stretch')
    st.caption('Frecuencias y porcentajes del archivo. No equivalen a prevalencias ponderadas de la población de Estados Unidos.')
    sexcounts=df.Sex.value_counts().reindex([0,1],fill_value=0)
    st.write(f'Sexo registrado en los datos validados: mujeres (Sex = 0): {sexcounts.loc[0]:,}; hombres (Sex = 1): {sexcounts.loc[1]:,}.')
    st.subheader('Revisión de calidad')
    st.write('Busca información faltante y valores fuera de los códigos o rangos esperados. Esto evalúa los datos, no la calidad de las predicciones.')
    st.dataframe(quality,hide_index=True,width='stretch')
    st.write(f'Filas excluidas por resultado ausente o inválido: {excluded:,}. Filas originales idénticas: {int(raw.duplicated().sum()):,}.')
    st.caption('Se conservan filas idénticas: pueden ser personas distintas. Los perfiles con las mismas 21 respuestas se mantienen en un mismo subconjunto para reducir fuga de información.')
    st.write('Los códigos inválidos se convierten en nulos. Si falta el resultado, se excluye la fila; los predictores faltantes se imputan al entrenar. BMI se valida como entero de 1 a 100: es una regla operativa de esta versión, no un umbral clínico.')
    st.subheader('Resumen EDA antes de imputar')
    st.caption('Describe cada columna validada. No se calcula la media de códigos categóricos. DS = desviación estándar; Q1/Q3 = cuartiles 25/75 %.')
    summary=eda(df)
    st.dataframe(summary,hide_index=True,width='stretch')
    st.download_button('Descargar EDA',summary.to_csv(index=False).encode('utf-8-sig'),'EDA.csv','text/csv')
    st.subheader('Distribución de una variable')
    st.write('Elige una columna para ver cuántas veces aparece cada valor. Es una descripción; no mide por sí sola la capacidad predictiva.')
    variable=st.selectbox('Variable a explorar',list(VARIABLES),format_func=lambda c:f'{nombre(c)} ({c})')
    st.write(VARIABLES[variable][1])
    st.caption('Códigos / unidades: '+VARIABLES[variable][2])
    if variable==TARGET: leyenda()
    freq=df[variable].value_counts().sort_index()
    if variable in ['BMI','MentHlth','PhysHlth']:
        chartdata=pd.DataFrame({'Valor':freq.index.astype(int).astype(str),'Registros':freq.to_numpy()})
        # Con muchos valores, agrupar en intervalos evita solapamiento de etiquetas.
        if len(freq)>20:
            intervals=pd.cut(df[variable].dropna(),bins=10,include_lowest=True)
            counted=intervals.value_counts(sort=False)
            chartdata=pd.DataFrame({'Valor':[str(i) for i in counted.index],'Registros':counted.to_numpy()})
            st.caption('Valores agrupados en 10 intervalos para facilitar lectura; consulta el detalle en la tabla.')
    else:
        chartdata=pd.DataFrame({'Valor':[str(int(v)) for v in freq.index],'Registros':freq.to_numpy(),'Significado':[significado(variable,v) for v in freq.index]})
    barras(chartdata,'Valor',nombre(variable))
    st.caption(f'Los nulos se excluyen del gráfico: {int(df[variable].isna().sum())} celdas sin valor.')
    st.dataframe(chartdata,hide_index=True,width='stretch')

with models:
    st.subheader('Cómo se elige el modelo')
    st.write('Se comparan regresión logística y bosque aleatorio con una referencia que siempre predice la clase mayoritaria. Las particiones aproximadas son 60 % entrenamiento, 20 % validación y 20 % prueba, manteniendo separados los perfiles iguales.')
    st.markdown('**F1 macro** = promedio del F1 de cada categoría. Combina precisión y recall, dando el mismo peso a las tres categorías aunque sus tamaños sean distintos.')
    st.latex(r'F1_k=\frac{2\,P_k\,R_k}{P_k+R_k},\qquad F1_{macro}=\frac{F1_0+F1_1+F1_2}{3}')
    leyenda()
    st.info('Regla de selección: entre los dos modelos predictivos se elige el de mayor F1 macro en validación, no necesariamente el de mayor exactitud. Luego se informa su desempeño en prueba. El modelo base queda como referencia; superar al otro candidato no garantiza superar una referencia o ser útil clínicamente.')
    st.caption('Si no hay aciertos/predicciones que permitan calcular un F1 por clase, se asigna 0 según la configuración de métricas. Los parámetros se mantienen fijos en esta revisión visual y explicativa.')
    if st.button('Entrenar y evaluar modelos',type='primary'):
        try:
            with st.spinner('Entrenando y evaluando; puede tardar varios minutos…'):
                st.session_state['resultado']=calcular(df)
                st.session_state.pop('perfil_predicho',None)
        except ValueError as exc:
            st.error(str(exc))
    result=st.session_state.get('resultado')
    if result:
        st.subheader('Composición de las particiones')
        splits=result['particiones'].copy()
        splits.index=[CLASES[int(v)] for v in splits.index]
        st.dataframe(splits,width='stretch')
        leyenda()
        st.subheader('Comparación en validación')
        st.caption('Exactitud y exactitud balanceada se muestran como porcentaje; F1 macro en escala 0–1.')
        st.dataframe(result['validacion'].style.format({'Exactitud':'{:.1%}','Exactitud balanceada':'{:.1%}','F1 macro':'{:.3f}'}),width='stretch')
        winner=result['ganador']
        winner_score=result['validacion'].loc[winner,'F1 macro']
        st.success(f'Modelo seleccionado: {winner}. F1 macro en validación: {winner_score:.3f}. Es el mayor de los dos candidatos predictivos.')
        st.subheader('Evaluación final en prueba')
        st.caption('Solo el modelo seleccionado. Son datos reservados; no deben usarse para ajustar repetidamente los parámetros.')
        a,b,c=st.columns(3)
        a.metric('Exactitud',f"{result['prueba']['Exactitud']:.1%}")
        b.metric('Exactitud balanceada',f"{result['prueba']['Exactitud balanceada']:.1%}")
        c.metric('F1 macro',f"{result['prueba']['F1 macro']:.3f}")
        st.write('Exactitud: proporción total de aciertos. Exactitud balanceada: promedio del recall de las tres categorías. F1 macro: promedio de sus F1; no es un porcentaje de aciertos.')
        report=result['reporte'].copy().drop(index='accuracy',errors='ignore')
        report=report.rename(index={**dict(zip(LABELS,CLASES.values())),'macro avg':'Promedio macro','weighted avg':'Promedio ponderado'},columns={'precision':'Precisión','recall':'Recall / sensibilidad','f1-score':'F1','support':'Registros reales'})
        st.dataframe(report.style.format({'Precisión':'{:.3f}','Recall / sensibilidad':'{:.3f}','F1':'{:.3f}','Registros reales':'{:.0f}'}),width='stretch')
        st.caption('Precisión: qué proporción de las predicciones de una clase es correcta. Recall: qué proporción de sus casos reales se detecta. F1: combina ambas. El promedio ponderado da más peso a las clases con más registros.')
        st.subheader('Matriz de confusión')
        st.write('Filas = categoría real; columnas = categoría predicha. La diagonal contiene aciertos y el resto, errores.')
        matrix=result['matriz'].copy()
        matrix.index=list(CLASES.values()); matrix.columns=list(CLASES.values())
        st.dataframe(matrix,width='stretch')
        leyenda()
        st.subheader('Variables que aportan información al modelo')
        importance=result['importancia'].copy()
        importance.insert(1,'Nombre en español',importance.Variable.map(nombre))
        st.dataframe(importance.rename(columns={'Caída media de F1 macro':'Pérdida de F1 macro'}),hide_index=True,width='stretch')
        # Se conserva la importancia calculada: F1 original menos F1 al desordenar.
        # Dos familias de color; el tono se intensifica al alejarse del cero.
        chartdata=importance.rename(columns={'Caída media de F1 macro':'Pérdida de F1 macro'}).copy()
        change=chartdata['Pérdida de F1 macro']
        lower=min(float(change.min()),-0.000001)
        upper=max(float(change.max()),0.000001)
        chartdata['Lectura']=change.map(lambda v: 'El F1 bajó al desordenar' if v>0 else ('El F1 subió al desordenar' if v<0 else 'Sin cambio medio'))
        st.markdown('**Cómo leer las barras:** azul a la derecha = el F1 bajó al desordenar la variable; naranja a la izquierda = el F1 subió. Cerca de cero = poco cambio.')
        bars=alt.Chart(chartdata).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X('Pérdida de F1 macro:Q',title='Pérdida de F1 macro al desordenar la variable',axis=alt.Axis(labelAngle=0,format='.3f')),
            y=alt.Y('Nombre en español:N',sort='-x',title=None,axis=alt.Axis(labelAngle=0,labelLimit=300)),
            color=alt.Color('Pérdida de F1 macro:Q',scale=alt.Scale(domain=[lower,0,upper],range=['#C2410C','#CBD5E1','#1D4ED8']),legend=None),
            tooltip=[alt.Tooltip('Variable:N'),alt.Tooltip('Nombre en español:N'),alt.Tooltip('Pérdida de F1 macro:Q',format='+.4f'),alt.Tooltip('DS entre repeticiones:Q',format='.4f'),alt.Tooltip('Lectura:N')])
        zero=alt.Chart(pd.DataFrame({'Cero':[0]})).mark_rule(color='#475569',strokeWidth=1.5).encode(x='Cero:Q')
        st.altair_chart((bars+zero).properties(height=600).configure_view(stroke=None),theme=None,width='stretch')
        st.caption('Pérdida = F1 original − F1 al desordenar. Cuanto más larga sea una barra AZUL hacia la derecha, más dependió el modelo de esa variable en esta evaluación. Acercarse a cero significa menor cambio.')
        with st.expander('Ver un ejemplo de interpretación'):
            st.write('Si el F1 pasa de 0,43 a 0,41, la pérdida es +0,02: barra azul a la derecha. Si pasa de 0,43 a 0,435, la pérdida es −0,005: barra naranja a la izquierda. Una barra negativa no significa que la variable proteja contra la diabetes.')
            st.write('Se usan hasta 3 000 registros de validación y tres repeticiones. La DS indica variación entre repeticiones. Las variables correlacionadas pueden compartir información; esta medida no demuestra causalidad.')
        st.download_button('Descargar comparación',result['validacion'].to_csv().encode('utf-8-sig'),'comparacion_modelos.csv','text/csv')
        st.download_button('Descargar importancia',importance.to_csv(index=False).encode('utf-8-sig'),'importancia_variables.csv','text/csv')
        st.warning('Revisa el recall de prediabetes y diabetes. Una exactitud alta puede coexistir con pocos casos detectados en las categorías minoritarias. Esta revisión no promete mejorar las métricas.')
    else:
        st.info('Presiona el botón para generar resultados con el archivo cargado.')

with simulation:
    st.subheader('Simulación académica de un perfil')
    st.write('Permite al analista comprobar qué categoría asigna el modelo a 21 respuestas. Los valores iniciales son un ejemplo fijo, no un paciente ni una recomendación.')
    st.warning('No pronostica que una persona vaya a desarrollar diabetes ni confirma su condición actual. No se presentan probabilidades como riesgo clínico.')
    result=st.session_state.get('resultado')
    if result:
        defaults={'BMI':25,'Age':5,'Education':4,'Income':5,'GenHlth':3,'PhysActivity':1,'Fruits':1,'Veggies':1,'AnyHealthcare':1,'CholCheck':1}
        with st.form('perfil'):
            header_name,header_input=st.columns([1.25,1],gap='large')
            header_name.markdown('**Variable**')
            header_input.markdown('**Respuesta del perfil**')
            profile={}
            for variable in FEATURES:
                label_col,input_col=st.columns([1.25,1],gap='large',vertical_alignment='center')
                with label_col:
                    st.markdown(f'**{nombre(variable)}**')
                    st.caption(variable)
                with input_col:
                    widget_label=f'{nombre(variable)} ({variable})'
                    if variable in BINARY:
                        profile[variable]=st.selectbox(widget_label,[0,1],index=defaults.get(variable,0),format_func=lambda v,col=variable:f'{v} · {significado(col,v)}',help=VARIABLES[variable][1],key='perfil_'+variable,label_visibility='collapsed')
                    elif variable in ['Age','Education','Income','GenHlth']:
                        lo,hi=RANGES[variable]
                        profile[variable]=st.selectbox(widget_label,list(range(lo,hi+1)),index=defaults.get(variable,lo)-lo,format_func=lambda v,col=variable:f'{v} · {significado(col,v)}',help=VARIABLES[variable][2],key='perfil_'+variable,label_visibility='collapsed')
                    else:
                        lo,hi=RANGES[variable]
                        profile[variable]=st.number_input(widget_label,min_value=lo,max_value=hi,value=defaults.get(variable,lo),step=1,help=VARIABLES[variable][2],key='perfil_'+variable,label_visibility='collapsed')
            submitted=st.form_submit_button('Clasificar perfil de ejemplo')
        if submitted:
            inputs=pd.DataFrame([profile],columns=FEATURES)
            code=int(result['modelo'].predict(inputs)[0])
            st.session_state['perfil_predicho']=(code,inputs)
        if 'perfil_predicho' in st.session_state:
            code,inputs=st.session_state['perfil_predicho']
            st.info('Categoría estimada por el modelo: '+CLASES[code])
            leyenda()
            st.caption('Resultado de la última clasificación enviada. Si cambias las entradas, vuelve a pulsar Clasificar.')
            st.dataframe(inputs,width='stretch')
    else:
        st.info('Primero entrena los modelos en «Predicción y resultados».')
