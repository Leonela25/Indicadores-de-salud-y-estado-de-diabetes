"""Textos y diccionario del CSV recodificado de Kaggle; no del archivo bruto CDC."""
CLASES = {0:'0 · Sin diabetes / solo durante el embarazo',1:'1 · Prediabetes',2:'2 · Diabetes'}
LEYENDA = 'Diabetes_012: 0 = sin diabetes o diabetes reportada solo durante el embarazo · 1 = prediabetes · 2 = diabetes.'
COLORES = ['#1D4ED8','#B45309','#15803D']
# nombre, definición de la pregunta/medida, códigos o unidades
VARIABLES = {
'Diabetes_012':('Estado de diabetes reportado','Resultado a clasificar, derivado de si alguna vez le indicaron que tenía diabetes. No distingue tipo 1 y tipo 2.','0 = sin diabetes o solo durante el embarazo; 1 = prediabetes; 2 = diabetes'),
'HighBP':('Presión arterial elevada','Indicador de haber recibido un reporte de presión arterial elevada. No es una medición de presión en mmHg.','0 = no; 1 = sí'),
'HighChol':('Colesterol elevado','Reporte de colesterol elevado indicado por un profesional. No es una concentración de laboratorio.','0 = no; 1 = sí'),
'CholCheck':('Control de colesterol','Se realizó un control de colesterol en los últimos cinco años.','0 = no; 1 = sí'),
'BMI':('Índice de masa corporal (IMC)','Relación entre peso y talla; no es peso en kilogramos.','kg/m²; valor numérico redondeado en el CSV'),
'Smoker':('Antecedente de consumo de cigarrillos','Ha fumado al menos 100 cigarrillos en toda su vida. No significa necesariamente que fuma actualmente.','0 = no; 1 = sí'),
'Stroke':('Antecedente de accidente cerebrovascular','Reporte de haber tenido un accidente cerebrovascular.','0 = no; 1 = sí'),
'HeartDiseaseorAttack':('Enfermedad coronaria o infarto','Indicador combinado de enfermedad coronaria o infarto de miocardio reportado.','0 = no; 1 = sí'),
'PhysActivity':('Actividad física','Actividad física o ejercicio durante los últimos 30 días, fuera del trabajo. No mide días de gimnasio.','0 = no; 1 = sí'),
'Fruits':('Consumo de frutas','Consumo de frutas al menos una vez al día.','0 = no; 1 = sí'),
'Veggies':('Consumo de vegetales','Consumo de vegetales al menos una vez al día.','0 = no; 1 = sí'),
'HvyAlcoholConsump':('Consumo elevado de alcohol','Indicador de consumo semanal elevado: más de 14 bebidas en hombres o más de 7 en mujeres, según la definición del BRFSS 2015. No es cualquier consumo.','0 = no; 1 = sí'),
'AnyHealthcare':('Cobertura de salud','Cuenta con alguna cobertura de atención de salud, incluidos seguros o programas públicos.','0 = no; 1 = sí'),
'NoDocbcCost':('Barrera económica de atención','En los últimos 12 meses necesitó consultar un médico y no pudo por el costo. Aquí sí = presencia de una barrera.','0 = no; 1 = sí'),
'GenHlth':('Salud general percibida','Valoración de su salud general. Un código más alto representa peor valoración.','1 = excelente; 2 = muy buena; 3 = buena; 4 = regular; 5 = mala'),
'MentHlth':('Días de mala salud mental','Días durante los últimos 30 en que su salud mental no fue buena.','0–30 días'),
'PhysHlth':('Días de mala salud física','Días durante los últimos 30 en que su salud física no fue buena.','0–30 días'),
'DiffWalk':('Dificultad para caminar','Dificultad seria para caminar o subir escaleras.','0 = no; 1 = sí'),
'Sex':('Sexo registrado','Codificación binaria del sexo en este archivo histórico. No es una clasificación de identidad de género.','0 = mujer; 1 = hombre. Estos códigos difieren del archivo bruto CDC.'),
'Age':('Grupo de edad','Edad agrupada: el código no es la edad en años.','1 = 18–24; 2 = 25–29; 3 = 30–34; 4 = 35–39; 5 = 40–44; 6 = 45–49; 7 = 50–54; 8 = 55–59; 9 = 60–64; 10 = 65–69; 11 = 70–74; 12 = 75–79; 13 = 80 o más'),
'Education':('Nivel educativo','Mayor nivel educativo completado según categorías de la encuesta estadounidense.','1 = sin escolaridad/solo kínder; 2 = grados 1–8; 3 = grados 9–11; 4 = secundaria completa/GED; 5 = 1–3 años de educación superior/técnica; 6 = 4 o más años de educación superior'),
'Income':('Categoría de ingreso anual del hogar','Ingreso anual del hogar en USD, según rangos de 2015. El código no es el ingreso monetario.','1 = menos de 10 000; 2 = 10 000–<15 000; 3 = 15 000–<20 000; 4 = 20 000–<25 000; 5 = 25 000–<35 000; 6 = 35 000–<50 000; 7 = 50 000–<75 000; 8 = 75 000 o más'),
}
AGE = {1:'18–24',2:'25–29',3:'30–34',4:'35–39',5:'40–44',6:'45–49',7:'50–54',8:'55–59',9:'60–64',10:'65–69',11:'70–74',12:'75–79',13:'80 o más'}
GEN = {1:'Excelente',2:'Muy buena',3:'Buena',4:'Regular',5:'Mala'}

def nombre(c):
    return VARIABLES[c][0]

def significado(c,v):
    v=int(v)
    if c=='Diabetes_012': return CLASES[v]
    if c=='Sex': return 'Mujer' if v==0 else 'Hombre'
    if c=='Age': return AGE[v]+' años'
    if c=='GenHlth': return GEN[v]
    if c in ['Income','Education']: return f'Código {v}; ver diccionario de esta variable'
    if c in ['BMI','MentHlth','PhysHlth']: return str(v)
    return 'No' if v==0 else 'Sí'

FUENTES = {
'CDC: qué es BRFSS':'https://www.cdc.gov/brfss/about/',
'CDC: documentación BRFSS 2015':'https://www.cdc.gov/brfss/annual_data/annual_2015.html',
'CDC: códigos originales 2015 (DIABETE3, SEX y variables fuente)':'https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf',
'Kaggle: archivo derivado de Alex Teboul':'https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset',
'Kaggle: notebook de preparación del autor':'https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook',
'scikit-learn: definición de F1 macro':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html',
}
