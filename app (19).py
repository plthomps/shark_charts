import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Shark Presence | Model Explorer', page_icon='🦈', layout='wide')

# Respect either existing Streamlit secret name. If neither is configured,
# preserve the existing public-dashboard behavior.
def password_gate():
    try:
        secret = st.secrets.get('dashboard_password', st.secrets.get('APP_PASSWORD', None))
    except FileNotFoundError:
        secret = None
    if not secret:
        return True
    if st.session_state.get('shark_authenticated', False):
        return True
    st.title('Shark research dashboard')
    with st.form('dashboard_login'):
        entered = st.text_input('Password', type='password')
        submitted = st.form_submit_button('Open dashboard')
    if submitted:
        import hmac
        if hmac.compare_digest(entered, str(secret)):
            st.session_state['shark_authenticated'] = True
            st.rerun()
        else:
            st.error('Incorrect password.')
    return False

if not password_gate():
    st.stop()
st.title('🦈 Cape Cod Shark Presence — Model Explorer')
st.caption('Explore held-out model performance, importance, SHAP, and saved one- and two-way PDP figures. No model retraining.')

# Each year is stored in a separate ZIP (<25 MB) under results/.
# A lightweight virtual archive provides the same namelist()/read() interface.
class MultiZip:
    def __init__(self, paths):
        self.zips = [zipfile.ZipFile(p) for p in paths]
        self.lookup = {}
        for z in self.zips:
            for name in z.namelist():
                self.lookup[name] = z
    def namelist(self):
        return list(self.lookup)
    def read(self, name):
        return self.lookup[name].read(name)

@st.cache_resource(show_spinner=False)
def open_archives():
    root = Path(__file__).parent / 'results'
    if not any(list(root.glob(f'{p}_*.zip')) for p in ('FORWARD','LOYO','RANDOM')):
        root = Path(__file__).parent  # Also supports ZIPs uploaded to repository root
    result = {}
    for protocol in ('FORWARD', 'LOYO', 'RANDOM'):
        paths = sorted(root.glob(f'{protocol}_*.zip')) + sorted((Path(__file__).parent if root != Path(__file__).parent else Path(__file__).parent / 'results').glob(f'{protocol}_*.zip'))
        if paths:
            result[protocol] = MultiZip(paths)
    return result

@st.cache_data(show_spinner=False)
def collect_metrics():
    out = []
    for protocol, z in open_archives().items():
        for name in z.namelist():
            if protocol == 'RANDOM' and re.fullmatch(r'(ALL|DAWN|DAY|DUSK|NIGHT)/metrics\.csv', name):
                try:
                    df = pd.read_csv(io.BytesIO(z.read(name)))
                    for row in df.to_dict('records'):
                        row['protocol'] = 'RANDOM'
                        row['group'] = name.split('/')[0]
                        row['test_year'] = 0
                        row['auc'] = row.get('roc_auc')
                        row['ap'] = row.get('average_precision')
                        out.append(row)
                except (ValueError, KeyError):
                    pass
            elif protocol != 'RANDOM' and name.endswith('/metrics.json'):
                try:
                    row = json.loads(z.read(name))
                    row.setdefault('protocol', protocol)
                    out.append(row)
                except (ValueError, KeyError):
                    pass
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def importance_table(kind):
    out = []
    for protocol, z in open_archives().items():
        for name in z.namelist():
            if not name.endswith('/' + kind + '.csv'):
                continue
            m = (re.search(r'/(\d{4})_(ALL|DAWN|DAY|DUSK|NIGHT)/', name) if protocol != 'RANDOM' else re.fullmatch(r'(ALL|DAWN|DAY|DUSK|NIGHT)/' + re.escape(kind) + r'\.csv', name))
            if not m:
                continue
            df = pd.read_csv(io.BytesIO(z.read(name)))
            if df.shape[1] < 2:
                continue
            if protocol == 'RANDOM':
                col = {'gini_importance':'importance','permutation_importance':'importance_mean','shap_importance':'mean_abs_shap'}.get(kind)
                if 'feature' not in df.columns or col not in df.columns: continue
                df = df[['feature',col]].copy()
            else:
                df = df.iloc[:, :2].copy()
            df.columns = ['feature', 'importance']
            df['protocol'] = protocol
            df['year'] = int(m.group(1)) if protocol != 'RANDOM' else 0
            df['group'] = m.group(2) if protocol != 'RANDOM' else m.group(1)
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['feature','importance','protocol','year','group'])

with st.sidebar:
    st.header('Results source')
    st.caption('Reading FORWARD, LOYO, and RANDOM ZIP archives from results/ (or repository root).')

try:
    archives = open_archives()
    metrics = collect_metrics()
except (zipfile.BadZipFile, ValueError) as e:
    st.error(f'Could not read results archives: {e}')
    st.stop()
if not archives:
    st.error('No ZIPs found. Add FORWARD_2018.zip, LOYO_2017.zip, RANDOM_ALL.zip, etc. to results/ or the repository root.')
    st.stop()

if metrics.empty:
    st.error('No model metrics found in the ZIP files.')
    st.stop()

with st.sidebar:
    st.divider()
    protocol=st.selectbox('Protocol',sorted(archives),index=0)
    metrics['test_year']=metrics.test_year.astype(int)
    available=metrics[metrics.protocol==protocol]
    group=st.selectbox('Light condition',['ALL','DAWN','DAY','DUSK','NIGHT'])
    year=(0 if protocol=='RANDOM' else st.selectbox('Held-out year',sorted(available[available.group==group].test_year.unique().tolist()),index=0))

metrics['test_year']=metrics.test_year.astype(int)
st.success(f"Loaded {len(metrics)} models — " + ', '.join(f"{p}: {sum(metrics.protocol==p)}" for p in archives))

def image_bytes(protocol,year,group,filename):
    z=archives.get(protocol)
    if not z:return None
    path=(f'{group}/{filename}' if protocol=='RANDOM' else f'{protocol}/{year}_{group}/{filename}')
    try: return z.read(path)
    except KeyError: return None

def image_panel(protocol,year,group,filename,caption=None):
    b=image_bytes(protocol,year,group,filename)
    if b is not None: st.image(b,caption=caption or f'{protocol} · {year} · {group} · {filename}',use_container_width=True)
    else: st.caption(f'Not generated: {filename}')

st.caption('RANDOM uses a stratified 80/20 split across all years; FORWARD and LOYO hold out years.')
tabs=st.tabs(['Compare all models','Overview','Model diagnostics','Feature importance','One-way PDPs','Two-way PDPs','Compare protocols','Interpretation guide'])

with tabs[0]:
    st.subheader('Compare model performance — all available models')
    st.caption('Each point is one model. RANDOM uses a stratified random test split; FORWARD and LOYO use held-out years. Compare AUCs with those different validation designs in mind.')
    compare = metrics.copy()
    compare['roc_auc_plot'] = pd.to_numeric(compare['auc'], errors='coerce')
    compare['accuracy_plot'] = pd.to_numeric(compare.get('accuracy', pd.Series(index=compare.index, dtype=float)), errors='coerce')
    compare['test_label'] = compare['test_year'].apply(lambda y: 'Pooled years' if int(y)==0 else str(int(y)))
    protocols = st.multiselect('Validation approaches', sorted(compare.protocol.unique()), default=sorted(compare.protocol.unique()), key='compare_protocols')
    groups = st.multiselect('Light conditions', ['ALL','DAWN','DAY','DUSK','NIGHT'], default=['ALL','DAWN','DAY','DUSK','NIGHT'], key='compare_groups')
    years = sorted(int(y) for y in compare.loc[compare.test_year>0,'test_year'].unique())
    selected_years = st.multiselect('Held-out years (FORWARD/LOYO)', years, default=years, key='compare_years')
    compare = compare[compare.protocol.isin(protocols) & compare.group.isin(groups) & ((compare.test_year==0) | compare.test_year.isin(selected_years))].copy()
    LIGHT_COLORS = {'ALL':'#4C566A','DAWN':'#E69F00','DAY':'#56B4E9','DUSK':'#CC79A7','NIGHT':'#0072B2'}
    if compare.empty:
        st.info('No models match the selected filters.')
    else:
        mode = st.radio('Chart view', ['Accuracy vs AUC (scatter)', 'AUC by year (lines)', 'AUC heatmap', 'AUC by light condition (3 methods)', 'All 80 AUCs together'], horizontal=True, key='compare_mode')
        hover = {'protocol':True,'group':True,'test_label':True,'n_test':True,'f1':':.3f','roc_auc_plot':':.3f','accuracy_plot':':.3f'}
        if mode == 'Accuracy vs AUC (scatter)':
            points = compare.dropna(subset=['accuracy_plot','roc_auc_plot'])
            if points.empty:
                st.warning('No models with both accuracy and AUC available.')
            else:
                fig = px.scatter(points, x='accuracy_plot', y='roc_auc_plot', color='group', symbol='protocol', color_discrete_map=LIGHT_COLORS,
                                 hover_name='test_label', hover_data=hover,
                                 labels={'accuracy_plot':'Accuracy','roc_auc_plot':'ROC AUC','protocol':'Validation','group':'Light condition'},
                                 title='Held-out model discrimination vs classification accuracy')
                fig.update_traces(marker={'size':11, 'line':{'width':0.8,'color':'rgba(30,30,30,.55)'}})
                fig.update_layout(height=620, legend_title_text='Validation / light condition')
                st.plotly_chart(fig, use_container_width=True)
        elif mode == 'AUC by year (lines)':
            yearly = compare[compare.test_year>0].dropna(subset=['roc_auc_plot'])
            if yearly.empty:
                st.info('Select FORWARD or LOYO and at least one year for this view.')
            else:
                fig = px.line(yearly.sort_values('test_year'), x='test_year', y='roc_auc_plot', color='group', color_discrete_map=LIGHT_COLORS,
                              line_dash='protocol', markers=True, hover_data=['n_test'],
                              labels={'test_year':'Held-out year','roc_auc_plot':'ROC AUC','protocol':'Validation','group':'Light condition'},
                              title='ROC AUC across held-out years')
                fig.update_layout(height=620)
                st.plotly_chart(fig, use_container_width=True)
            random_rows=compare[compare.protocol=='RANDOM']
            if not random_rows.empty:
                st.caption('RANDOM results are pooled across years, so they are shown in the scatter and table, not assigned to a single year.')
        elif mode == 'AUC heatmap':
            yearly = compare[compare.test_year>0].dropna(subset=['roc_auc_plot']).copy()
            if yearly.empty:
                st.info('The heatmap requires FORWARD or LOYO models.')
            else:
                for method in sorted(yearly.protocol.unique()):
                    grid = yearly[yearly.protocol==method].pivot_table(index='group',columns='test_year',values='roc_auc_plot').reindex(['ALL','DAWN','DAY','DUSK','NIGHT'])
                    grid=grid.dropna(how='all')
                    if not grid.empty:
                        fig=px.imshow(grid, text_auto='.3f', aspect='auto', color_continuous_scale='Viridis',
                                      labels={'x':'Held-out year','y':'Light condition','color':'ROC AUC'},title=f'{method} — ROC AUC')
                        st.plotly_chart(fig, use_container_width=True)
        elif mode == 'AUC by light condition (3 methods)':  # Every AUC, faceted by validation approach
            points = compare.dropna(subset=['roc_auc_plot']).copy()
            if points.empty:
                st.info('No AUC values match the selected filters.')
            else:
                points['protocol'] = pd.Categorical(points['protocol'], categories=['RANDOM','FORWARD','LOYO'], ordered=True)
                points['group'] = pd.Categorical(points['group'], categories=['ALL','DAWN','DAY','DUSK','NIGHT'], ordered=True)
                fig = px.strip(points, x='group', y='roc_auc_plot', color='group',
                               facet_col='protocol', facet_col_spacing=0.045,
                               stripmode='overlay', color_discrete_map=LIGHT_COLORS,
                               category_orders={'group':['ALL','DAWN','DAY','DUSK','NIGHT'],
                                                'protocol':['RANDOM','FORWARD','LOYO']},
                               hover_name='test_label',
                               hover_data={'protocol':True,'group':True,'roc_auc_plot':':.3f','n_test':True},
                               labels={'group':'Light condition','roc_auc_plot':'ROC AUC','protocol':'Validation method'},
                               title='Individual model AUCs by validation method and light condition')
                fig.update_traces(marker={'size':10, 'opacity':0.82,
                                          'line':{'width':0.6,'color':'rgba(30,30,30,.5)'}},
                                  jitter=0.48)
                fig.update_layout(height=590, legend_title_text='Light condition')
                fig.update_yaxes(range=[0.45,1.0])
                fig.for_each_annotation(lambda a: a.update(text=a.text.replace('protocol=', '')))
                st.plotly_chart(fig, use_container_width=True)
                st.caption('Each dot represents one model; hover for the held-out year (or pooled RANDOM), AUC, and test size. No medians or averages are used. RANDOM uses a pooled-year random split, unlike the held-out-year FORWARD and LOYO tests.')
        else:  # All individual AUCs on one combined chart
            points = compare.dropna(subset=['roc_auc_plot']).copy()
            if points.empty:
                st.info('No AUC values match the selected filters.')
            else:
                fig = px.strip(points, x='protocol', y='roc_auc_plot', color='group',
                               custom_data=['protocol'],
                               stripmode='group', color_discrete_map=LIGHT_COLORS,
                               category_orders={'protocol':['RANDOM','FORWARD','LOYO'],
                                                'group':['ALL','DAWN','DAY','DUSK','NIGHT']},
                               hover_name='test_label',
                               hover_data={'protocol':True,'group':True,'roc_auc_plot':':.3f',
                                           'n_test':True},
                               labels={'protocol':'Validation method','roc_auc_plot':'ROC AUC',
                                       'group':'Light condition'},
                               title='All individual model AUCs — three methods, five light conditions')
                # Symbols distinguish validation protocols; colors distinguish light conditions.
                # px.strip groups points by light condition, so assign symbols per observation.
                method_symbols = {'RANDOM': 'circle', 'FORWARD': 'triangle-up', 'LOYO': 'square'}
                for trace in fig.data:
                    protocols = [row[0] for row in trace.customdata]
                    trace.marker.symbol = [method_symbols.get(method, 'circle') for method in protocols]
                    trace.marker.size = 10
                    trace.marker.opacity = 0.8
                    trace.marker.line = {'width': 0.5, 'color': 'rgba(30,30,30,.5)'}
                    trace.jitter = 0.35
                fig.update_layout(height=640, legend_title_text='Light condition')
                st.caption('Symbols: ● RANDOM (circle) · ▲ FORWARD (triangle) · ■ LOYO (square). Colors indicate light condition.')
                fig.update_yaxes(range=[0.45,1.0])
                st.plotly_chart(fig, use_container_width=True)
                st.caption('Every dot is one model (up to 80). Colors indicate light condition. '
                           'RANDOM pools years, while FORWARD and LOYO test held-out years; '
                           'their AUCs should not be interpreted as directly equivalent.')
        st.markdown('#### Model results table')
        columns=[c for c in ['protocol','group','test_label','auc','accuracy','ap','f1','n_test','n_train'] if c in compare.columns]
        st.dataframe(compare[columns].sort_values('auc',ascending=False),use_container_width=True,hide_index=True)
        st.download_button('Download comparison CSV', data=compare[columns].to_csv(index=False).encode('utf-8'),
                           file_name='shark_model_performance_comparison.csv',mime='text/csv')
    st.caption('Caution: accuracy depends on the classification threshold and class balance. ROC AUC summarizes discrimination across thresholds. Random splits can overestimate generalization when nearby observations are temporally correlated.')

with tabs[1]:
    st.subheader('Held-out performance — selected model')
    selected = metrics[(metrics.protocol == protocol) & (metrics.test_year == year) & (metrics.group == group)]
    if not selected.empty:
        r = selected.iloc[0]
        cols = st.columns(4)
        for col, (field, label) in zip(cols, [('auc','AUC'),('ap','Average precision'),('balanced_accuracy','Balanced accuracy'),('recall','Recall')]):
            if field in r and pd.notna(r[field]): col.metric(label, f"{float(r[field]):.3f}")
        st.dataframe(selected.T, use_container_width=True)
    with st.expander('Compare performance across years and light conditions'):
        measure=st.selectbox('Metric',['auc','ap','balanced_accuracy','recall','specificity','f1','precision','accuracy'],index=0)
        m=metrics[(metrics.protocol==protocol)&(metrics.group==group)].copy()
        if measure not in m.columns: st.info('Metric unavailable for this protocol.')
        elif protocol=='RANDOM':
            fig=px.bar(metrics[metrics.protocol=='RANDOM'],x='group',y=measure,title=f'RANDOM — {measure.upper()} by light condition')
            fig.update_yaxes(range=[0,1]);st.plotly_chart(fig,use_container_width=True)
        else:
            fig=px.line(m,x='test_year',y=measure,markers=True,title=f'{protocol} · {group}: {measure.upper()} by held-out year')
            fig.update_yaxes(range=[0,1]);st.plotly_chart(fig,use_container_width=True)
            pivot=metrics[metrics.protocol==protocol].pivot(index='group',columns='test_year',values=measure)
            st.plotly_chart(px.imshow(pivot,aspect='auto',color_continuous_scale='Viridis',zmin=0,zmax=1,text_auto='.2f',title=f'{protocol} — all years and groups'),use_container_width=True)

with tabs[2]:
    st.subheader('Saved model diagnostics')
    if year is None: st.info('Select a year in the sidebar.')
    else:
        diag=st.selectbox('Figure',(['confusion_matrix.png','roc_curve.png','precision_recall.png','gini_importance.png','permutation_importance.png','shap_summary.png'] if protocol=='RANDOM' else ['confusion_matrix.png','roc.png','gini.png','permutation.png','shap_summary.png']))
        image_panel(protocol,year,group,diag)
        selected=metrics[(metrics.protocol==protocol)&(metrics.test_year==year)&(metrics.group==group)]
        if not selected.empty: st.dataframe(selected.T, use_container_width=True)

with tabs[3]:
    st.subheader(f'Feature importance — {protocol} · {year} · {group}')
    kind=st.selectbox('Importance measure',['gini_importance','permutation_importance','shap_importance'],format_func=lambda k: {'gini_importance':'Gini (impurity)','permutation_importance':'Permutation (held-out)','shap_importance':'Mean absolute SHAP'}[k])
    importance=importance_table(kind)
    part=importance[(importance.protocol==protocol)&(importance.group==group)&(importance.year==year)].copy()
    if part.empty: st.warning('No importance CSV available for this model.')
    else:
        part=part.sort_values('importance',ascending=False)
        top=st.slider('Number of features',1,len(part),min(15,len(part)))
        shown=part.head(top).sort_values('importance')
        fig=px.bar(shown,x='importance',y='feature',orientation='h',title=f'{protocol} · {year} · {group} · {kind.replace("_"," ")}')
        fig.update_layout(height=max(420,top*30+140),yaxis_title='',xaxis_title='Importance')
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(part[['feature','importance']],use_container_width=True,hide_index=True)
        if kind=='shap_importance':
            st.markdown('#### SHAP summary (beeswarm)')
            image_panel(protocol,year,group,'shap_summary.png')
        if protocol!='RANDOM':
          with st.expander('Compare this importance measure across years'):
            all_part=importance[(importance.protocol==protocol)&(importance.group==group)]
            chosen=part.head(top).feature.tolist()
            grid=all_part[all_part.feature.isin(chosen)].pivot_table(index='feature',columns='year',values='importance').reindex(chosen)
            st.plotly_chart(px.imshow(grid,aspect='auto',color_continuous_scale='Viridis',title=f'{protocol} · {group} · across years'),use_container_width=True)
    st.caption('Gini, permutation, and mean absolute SHAP measure different properties; their scales are not directly comparable.')

with tabs[4]:
    st.subheader('One-way partial dependence')
    st.caption('These are averaged model predictions, not observed detection rates. RANDOM has individual one-way PDP images.')
    if protocol=='RANDOM':
        choices=sorted(n.split('/')[-1] for n in archives['RANDOM'].namelist() if re.fullmatch(rf'{group}/pdp_1d_.*\.png',n))
        if choices: image_panel(protocol,year,group,st.selectbox('One-way PDP',choices))
        else: st.info('No one-way PDP images found for this group.')
    elif year is not None: image_panel(protocol,year,group,'pdp_1way.png')
    else: st.info('Select a year in the sidebar.')
    if protocol!='RANDOM':
        with st.expander('Compare all years for this light condition'):
            for yr in sorted(available[available.group==group].test_year.unique()):
                st.markdown(f'**{protocol} — {yr} — {group}**')
                image_panel(protocol,int(yr),group,'pdp_1way.png')

with tabs[5]:
    st.subheader(f'Two-way PDP contours — {protocol} · {year} · {group}')
    st.caption('Contours show average model predictions, not observed detection rates or causal effects.')
    z=archives[protocol]
    pattern=(re.compile(rf'^{group}/(pdp_2d_.*\.png)$') if protocol=='RANDOM' else re.compile(rf'^{protocol}/{year}_{group}/(pdp2_.*\.png)$'))
    pairs=sorted({m.group(1) for n in z.namelist() if (m:=pattern.match(n))})
    if not pairs: st.warning('No two-way PDP images found for this selected model.')
    else:
        selected_pair=st.selectbox('Interaction (two environmental variables)',pairs,format_func=lambda t:t.removeprefix('pdp2_').removeprefix('pdp_2d_').removesuffix('.png').replace('__',' × ').replace('_vs_',' × '))
        image_panel(protocol,year,group,selected_pair)
        if protocol!='RANDOM':
          with st.expander('Compare the same interaction across years'):
            pattern_all=re.compile(rf'^{protocol}/(\d{{4}})_{group}/(pdp2_.*\.png)$')
            entries={(int(m.group(1)),m.group(2)) for n in z.namelist() if (m:=pattern_all.match(n))}
            years=sorted({yr for yr,fn in entries if fn==selected_pair})
            comparison_years=st.multiselect('Comparison years',years,default=[])
            for yr in comparison_years: image_panel(protocol,yr,group,selected_pair)

with tabs[6]:
    st.subheader('FORWARD versus LOYO')
    if not {'FORWARD','LOYO'}.issubset(archives): st.info('Upload both FORWARD and LOYO ZIPs for comparison.')
    elif protocol=='RANDOM': st.info('Choose FORWARD or LOYO in the sidebar to compare held-out years.')
    elif year is None: st.info('Select a year in the sidebar.')
    else:
        mode=st.radio('Compare',['One-way PDP','ROC','Confusion matrix','SHAP summary'],horizontal=True)
        filename={'One-way PDP':'pdp_1way.png','ROC':'roc.png','Confusion matrix':'confusion_matrix.png','SHAP summary':'shap_summary.png'}[mode]
        cols=st.columns(2)
        for col,p in zip(cols,['FORWARD','LOYO']):
            with col:
                st.markdown(f'**{p} — {year} — {group}**')
                image_panel(p,year,group,filename)
        st.caption('For 2024, the two protocols share the same training years; matching model outputs are expected if implementation is identical.')

with tabs[7]:
    st.subheader('Reading PDPs responsibly')
    st.markdown('''
- **Inside a two-way PDP:** contour labels (e.g., 0.45–0.55) are *averaged model outputs*. The X/Y axes are predictor levels.
- **A maximum of 0.53 is not a shark detection.** Compare the modeled range and curve/surface shape, not just whether it crosses 0.50.
- **Model probabilities may be uncalibrated**, especially with class weighting. Do not equate 0.53 with a measured 53% detection frequency.
- **PDPs are not causal effects.** Correlated predictors can yield implausible feature combinations; support for the joint X/Y region matters.
- **Two-way variation is not automatically interaction.** An additive combination of two main effects can also produce a varying contour surface.
- **Model generalization matters:** inspect held-out AUC, prevalence, and sample size before trusting year-specific patterns.
- **Acoustic detection ≠ complete biological presence.** Interpret the modeled target as detected presence.
    ''')
    st.markdown('#### Suggested research workflow')
    st.write('Start with FORWARD / ALL across years, compare the highest-ranked predictors, inspect corresponding one- and two-way PDPs, then check NIGHT and DAY. Use LOYO as a sensitivity comparison. Record the direction, approximate range, and consistency of each relationship.')
