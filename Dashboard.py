from google.cloud import bigquery
import os
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import ast
import pandas as pd
import statsmodels


#PAGE CONFIG

st.set_page_config(
    page_title="Greetings",
    page_icon="👋",
    layout = 'wide'
)

# READING AND PROCESS THE DATA FROM THE CLOUD
credential_filename = 'read-write-bq.json'
query = """
SELECT *
FROM (table)
"""
st.cache_data()
def read_data(credential_filename,query):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_filename
    client = bigquery.Client()
    results = client.query(query)
    return results.to_dataframe()

def output_to_list(output):
    start = output.find("[")  # Find the starting bracket
    end = output.rfind("]")
    python_list = ast.literal_eval(output[start:end+1])
    return python_list


if "df" not in st.session_state.keys():
    with st.spinner('Reading Data from Cloud'):
        labels = pd.read_csv('topic_labels.csv')
        df = read_data(credential_filename,query)
        df['tools'] = df['tools'].apply(lambda x : [y.lower() for y in output_to_list(x)])
        df['education'] = df['education'].apply(lambda x : [y.lower() for y in output_to_list(x)])
        df['responsibilities']  = df['responsibilities'].apply(lambda x :[y.lower() for y in output_to_list(x)])
        df['yoe'] = df['yoe'].apply(lambda x : output_to_list(x)[0] if len(output_to_list(x)) > 0 else None)
        df['locations'] = df['locations'].apply(lambda x: x.split(','))
        df = pd.concat([df,labels],axis=1).rename(columns={'0':'labels'})
        st.session_state.df = df 
else:
    df = st.session_state.df


loc_list = []
for i in list(df['locations']):
    loc_list = list(set(loc_list).union(set(i)))
tool_list = []
for i in list(df['tools']):
    tool_list = list(set(tool_list).union(set(i)))
educ_list = []
for i in list(df['education']):
    educ_list = list(set(educ_list).union(set(i)))
label_list = list(df['labels'].unique())

loc_list.remove('philippines')
min_min_salary = df['min_salary'].min()
max_max_salary = df['max_salary'].max()
min_yoe = df['yoe'].min()
max_yoe = df['yoe'].max()

#SIDEBAR FILTERS
with st.sidebar:
    st.write('### Filters')
    st.write('#### Categorical filters')
    st.write('These will select records that satisfy at least on of the selections per filter.')
    locations = st.multiselect('Filter based on location',loc_list)
    toolset = st.multiselect('Filter based on toolset',tool_list)
    educset = st.multiselect('Filter based on educational background',educ_list)
    labelset = st.multiselect('Filter based on job description topic (See Job Description Clustering page)',label_list)



    st.divider()
    st.write('#### Numerical filters')
    on1 = st.toggle("Scroller for salary filtering?",value=True)
    if on1:
        minimum_salary = st.slider('Salary Lower Bound',min_min_salary,max_max_salary)
        maximum_salary = st.slider('Salary Upper Bound',minimum_salary,max_max_salary,value=max_max_salary)
    else:
        minimum_salary = st.number_input('Salary Lower Bound',df['min_salary'].min(),max_max_salary)
        maximum_salary = st.number_input('Salary Upper Bound',minimum_salary,max_max_salary,value=max_max_salary)
    on2 = st.toggle("Scroller for YoE filtering?",value=True)
    if on2:
        minimum_yoe = st.slider('YoE Lower Bound',min_yoe,max_yoe)
        maximum_yoe = st.slider('Yoe Upper Bound',min_yoe,max_yoe,value=max_yoe)
    else:
        minimum_yoe = st.number_input('YoE Lower Bound',min_yoe,max_yoe)
        maximum_yoe = st.number_input('Yoe Upper Bound',min_yoe,max_yoe,value=max_yoe)


#APPLY FILTERS

df = df[ (df['min_salary'].isna()) | ((df['min_salary']>= minimum_salary) & (df['max_salary']<=maximum_salary))]
df = df[ (df['yoe'].isna()) | ((df['yoe']>= minimum_yoe) & (df['yoe']<=maximum_yoe))] 
if locations!=[]:  
    df = df.loc[[ len(set(locations).intersection(set(i)))!=0 for i in list(df['locations'])],:]
if toolset!=[]:
    df = df.loc[[ len(set(toolset).intersection(set(i)))!=0 for i in list(df['tools'])],:]
if educset!=[]:
    df = df.loc[[len(set(educset).intersection(set(i)))!=0 for i in list(df['education'])],:]
if labelset!=[]:
    df = df.loc[[i in labelset for i in df['labels']]]

#MAIN FUNCTIONS FOR PLOTS
#Metrics
def no_jobs(df):
    return len(df)

def most_hiring(df):
    return df['companies'].value_counts() if len(df)>0 else None

def mean_median_min_sal(df):
    rounded_min_salary_median = round(df['min_salary'].median(),0)
    rounded_min_salary_mean = round(df['min_salary'].mean(),0)
    return rounded_min_salary_median , rounded_min_salary_mean if len(df)>0 else None

def mean_median_max_sal(df):
    return round(df['max_salary'].median(),0), round(df['max_salary'].mean(),0) if len(df)>0 else None

def common_tool(df):
    return df.explode(['tools'])['tools'].value_counts()

def common_skills(df):
    return df.explode(['responsibilities'])['responsibilities'].value_counts() if len(df)>0 else None

def mean_median_yoe(df):
    return round(df['yoe'].median(),0), round(df['yoe'].mean(),0) if len(df)>0 else None

def common_educ(df): 
    return df.explode(['education'])['education'].value_counts() if len(df)>0 else None


#PLOTS

def ave_sal_dist(df):
    ave_sal = (df['min_salary'] + df['max_salary'])/2
    fig = px.histogram(ave_sal,
                       labels={'value': "Averaged Salary"}
                       )
    fig.update(layout_showlegend=False)
    return fig, ave_sal

def salary_density_heatmap(df):
    fig = px.density_heatmap(
        df, 
        x = 'min_salary', y = 'max_salary',
        text_auto= True, 
        marginal_x='histogram', marginal_y='histogram',
        nbinsx = 15, nbinsy =15)
    return fig, df['min_salary'], df['max_salary']

def common_tool_plot(common_tool_df):
    common_tool_df = common_tool_df.reset_index().loc[:10,:]
    fig = px.bar(common_tool_df, x='tools', y='count')
    return fig

def common_skills_plot(common_skills_df):
    common_skills_df = common_skills_df.reset_index().loc[:10,:]
    fig = px.bar(common_skills_df, x='responsibilities', y='count')
    return fig

def common_educ_plot(common_educ_df):
    common_educ_df = common_educ_df.reset_index().loc[:8,:]
    fig = px.bar(common_educ_df, x='education', y='count')
    return fig

def yoe_hist(df):
    fig = px.histogram(df, x = 'yoe')
    return fig, df['yoe']

def yoe_min_sal_scatter(df):
    fig = px.scatter(df, x='yoe', y='min_salary', trendline='ols')
    return fig

model1 = px.get_trendline_results(yoe_min_sal_scatter(df))

def yoe_max_sal_scatter(df):
    fig = px.scatter(df, x='yoe', y='max_salary', trendline='ols')
    return fig
model2 = px.get_trendline_results(yoe_max_sal_scatter(df))

def work_type_ring(df):
    df_ring = df.work_types.value_counts().reset_index()
    fig = px.pie(df_ring, values='count', names = 'work_types', hole = 0.3)
    return fig



# PAGE STRUCTURE

st.write("""# Data Analyst Job Statistics Dashboard""")

st.divider()

met1, met2, met3, met4, met5, met6 = st.columns(6)

st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 30px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

with met1:
    st.metric('Median Minimum Salary',f'₱ {mean_median_min_sal(df)[0]}')
    st.metric('Mean Minimum Salary',f'₱ {mean_median_min_sal(df)[1]}')
with met2:
    st.metric('Median Maximum Salary',f'₱ {mean_median_max_sal(df)[0]}')
    st.metric('Mean Maximum Salary',f'₱ {mean_median_max_sal(df)[1]}')
with met3:
    try:
        st.metric('Most Common Technologies',common_tool(df).index[0])
    except:
        st.metric('Most Common Technologies',None)
    try:
        st.metric('Most Common Skill/Responsibility',common_skills(df).index[0])
    except:
        st.metric('Most Common Skill/Responsibility',None)
with met4:
    st.metric('Median Years of Experience (YoE)',mean_median_yoe(df)[0])
    st.metric('Mean Years of Experience (YoE)',mean_median_yoe(df)[1])
with met5:
    try:
        st.metric('Most Desirable Educational Background',f' {common_educ(df).index[0]}')
    except:
        st.metric('Most Desirable Educational Background',None)
with met6:
    st.metric('No. of Jobs in Database',no_jobs(df))

st.divider()
st.write("""Hover over the plots for more detail. Access the sidebar through the button on the top left to apply filters on the dashboard.""")
st.write("""Note: Given a salary range: ₱ 30000 - ₱40000, 
         the minimum salary is ₱ 30000, the maximum salary is ₱ 40000, and the averaged salary is the average, which is ₱ 35000. """)
box1, box2 = st.columns(2)

with box1:
    st.write('## Averaged Salary Distribution')
    st.plotly_chart(ave_sal_dist(df)[0])
    with st.popover('Info'):
        min_ave_sal = ave_sal_dist(df)[1].min()
        max_ave_sal = ave_sal_dist(df)[1].max()
        st.write(f'Minimum Averaged Salary: {min_ave_sal}')
        st.write(f'Maximum Averaged Salary: {max_ave_sal}')
        st.write(f'Range: {max_ave_sal-min_ave_sal}')

with box2:
    st.write('## Salary Range Heatmap')
    st.plotly_chart(salary_density_heatmap(df)[0])
    with st.popover('Info'):
        small_min_sal = salary_density_heatmap(df)[1].min()
        larg_min_sal = salary_density_heatmap(df)[1].max()
        st.write(f'Smallest Minimum Salary: {small_min_sal}')
        st.write(f'Largest Minimum Salary: {larg_min_sal }')
        st.write(f'Range: {larg_min_sal-small_min_sal}')
        st.divider()
        small_max_sal = salary_density_heatmap(df)[2].min()
        larg_max_sal = salary_density_heatmap(df)[2].max()
        st.write(f'Minimum Averaged Salary: {small_max_sal}')
        st.write(f'Maximum Averaged Salary: {larg_max_sal}')
        st.write(f'Range: {larg_max_sal-small_max_sal}')

st.divider()

box3, box4, box5 = st.columns(3)

with box3:
    st.write('## Most Desired Technologies')
    st.plotly_chart(common_tool_plot(common_tool(df)))

with box4:
    st.write('## Most Desired Skills')
    st.plotly_chart(common_skills_plot(common_skills(df)))
with box5:
    st.write("## Most Desired Education")
    st.plotly_chart(common_educ_plot(common_educ(df)))

st.divider()

box6, box7, box8 = st.columns(3)

with box6:
    st.write('## YoE Distribution')
    st.plotly_chart(yoe_hist(df)[0])
    with st.popover('Info'):
        min_yoe = yoe_hist(df)[1].min()
        max_yoe = yoe_hist(df)[1].max()
        st.write(f'Minimum YoE: {min_yoe}')
        st.write(f'Maximum YoE: { max_yoe }')
        st.write(f'Range: { max_yoe-min_yoe}')

with box7:
    st.write('## YoE vs. Minimum Salary Trend')
    st.plotly_chart(yoe_min_sal_scatter(df))
    #Tooltips
    with st.popover('Regression Results'):
        if len(model1)>0:
            p_value_1 = model1.iloc[0]['px_fit_results'].pvalues[0]
            r2_adjusted_1 = model1.iloc[0]['px_fit_results'].rsquared_adj
            sig =  p_value_1<0.05
            beta0 = model1.iloc[0]['px_fit_results'].params[0]
            beta1 = model1.iloc[0]['px_fit_results'].params[1]
            st.write("The regression model is:")
            st.write(r''' min \_ salary = ''' + str(round(beta1,2))+  r'''*YoE + ''' + str(round(beta0,2)))
            st.write('The p-value resulting from testing if "a" in the model ')
            st.latex(r''' min \_ salary = a*YoE + b ''')
    
            st.write(f''' is not zero is {round(p_value_1,4)}.
                    Since, {str(round(p_value_1,4)) + ' < 0.05' if sig else str(round(p_value_1,4)) + ' > 0.05 ' }, there is { "" if sig else ' no '} evidence that 'YoE' has 
                    a linear relationship with min_salary.''')
            
            st.write(f'''Also, the coefficient of determination is {round(r2_adjusted_1,4)},
                    meaning YoE explains about {round(r2_adjusted_1*100,2)}% of the variability in min_salary.''')
        else:
            st.write('No data.')
        

with box8:
    st.write('## YoE vs. Maximum Salary Trend')
    st.plotly_chart(yoe_max_sal_scatter(df))
    #Tooltips
    with st.popover('Regression Results'):
        if len(model2)>0:
            p_value_2 = model2.iloc[0]['px_fit_results'].pvalues[0]
            r2_adjusted_2 = model2.iloc[0]['px_fit_results'].rsquared_adj
            sig =  p_value_2<0.05
            beta0 = model2.iloc[0]['px_fit_results'].params[0]
            beta1 = model2.iloc[0]['px_fit_results'].params[1]
            st.write("The regression model is:")
            st.write(r''' max \_ salary = ''' + str(round(beta1,2))+  r'''*YoE + ''' + str(round(beta0,2)))
            st.write('The p-value resulting from testing if "a" in the model ')
            st.latex(r''' max \_ salary = a*YoE + b ''')
    
            st.write(f''' is not zero is {round(p_value_2,4)}.
                    Since, {str(round(p_value_2,4)) + ' < 0.05' if sig else str(round(p_value_2,4)) + ' > 0.05 ' }, there is { "" if sig else ' no '} evidence that 'YoE' has 
                    a linear relationship with min_salary.''')
            
            st.write(f'''Also, the coefficient of determination is {round(r2_adjusted_2,4)},
                    meaning YoE explains about {round(r2_adjusted_2*100,2)}% of the variability in min_salary.''')
        else:
            st.write('No data.')









