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
    result = {}
    for protocol in ('FORWARD', 'LOYO'):
        paths = sorted(root.glob(f'{protocol}_*.zip'))
        if paths:
            result[protocol] = MultiZip(paths)
    return result

@st.cache_data(show_spinner=False)
def collect_metrics():
    out = []
    for protocol, z in open_archives().items():
        for name in z.namelist():
            if name.endswith('/metrics.json'):
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
            m = re.search(r'/(\d{4})_(ALL|DAWN|DAY|DUSK|NIGHT)/', name)
            if not m:
                continue
            df = pd.read_csv(io.BytesIO(z.read(name)))
            if df.shape[1] < 2:
                continue
            df = df.iloc[:, :2].copy()
            df.columns = ['feature', 'importance']
            df['protocol'] = protocol
            df['year'] = int(m.group(1))
            df['group'] = m.group(2)
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['feature','importance','protocol','year','group'])

with st.sidebar:
    st.header('Results source')
    st.caption('Reading per-year ZIP archives from the results/ folder in this GitHub repository.')

try:
    archives = open_archives()
    metrics = collect_metrics()
except (zipfile.BadZipFile, ValueError) as e:
    st.error(f'Could not read results archives: {e}')
    st.stop()
if not archives:
    st.error('No ZIPs found. Add results/FORWARD_2018.zip, etc. to the repository.')
    st.stop()

if metrics.empty:
    st.error('No metrics.json files found. The ZIPs should contain FORWARD/2018_ALL/metrics.json or LOYO/2017_ALL/metrics.json, etc.')
    st.stop()

with st.sidebar:
    st.divider()
    protocol=st.selectbox('Protocol',sorted(archives),index=0)
    metrics['test_year']=metrics.test_year.astype(int)
    available=metrics[metrics.protocol==protocol]
    group=st.selectbox('Light condition',['ALL','DAWN','DAY','DUSK','NIGHT'])
    year=st.selectbox('Held-out year',sorted(available[available.group==group].test_year.unique().tolist()),index=None,placeholder='Choose a year')

metrics['test_year']=metrics.test_year.astype(int)
st.success(f"Loaded {len(metrics)} models — " + ', '.join(f"{p}: {sum(metrics.protocol==p)}" for p in archives))

def image_bytes(protocol,year,group,filename):
    z=archives.get(protocol)
    if not z:return None
    path=f'{protocol}/{year}_{group}/{filename}'
    try: return z.read(path)
    except KeyError: return None

def image_panel(protocol,year,group,filename,caption=None):
    b=image_bytes(protocol,year,group,filename)
    if b is not None: st.image(b,caption=caption or f'{protocol} · {year} · {group} · {filename}',use_container_width=True)
    else: st.caption(f'Not generated: {filename}')

tabs=st.tabs(['Overview','Model diagnostics','Feature importance','One-way PDPs','Two-way PDPs','Compare protocols','Interpretation guide'])
with tabs[0]:
    st.subheader('Held-out performance across years')
    m=metrics[metrics.group==group].copy()
    a,b,c=st.columns(3)
    a.metric('Models',len(metrics));b.metric('Forward models',sum(metrics.protocol=='FORWARD'));c.metric('LOYO models',sum(metrics.protocol=='LOYO'))
    measure=st.selectbox('Metric',['auc','ap','balanced_accuracy','recall','specificity','f1','precision','accuracy'],index=0)
    fig=px.line(m,x='test_year',y=measure,color='protocol',markers=True,title=f'{group}: {measure.upper()} by held-out year')
    fig.update_yaxes(range=[0,1]);st.plotly_chart(fig,use_container_width=True)
    pivot=metrics[metrics.protocol==protocol].pivot(index='group',columns='test_year',values=measure)
    fig=px.imshow(pivot,aspect='auto',color_continuous_scale='Viridis',zmin=0,zmax=1,text_auto='.2f',title=f'{protocol} — {measure.upper()}')
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(metrics.sort_values(['protocol','test_year','group']),use_container_width=True,hide_index=True)

with tabs[1]:
    st.subheader('Saved model diagnostics')
    if year is None: st.info('Select a year in the sidebar.')
    else:
        diag=st.selectbox('Figure',['confusion_matrix.png','roc.png','gini.png','permutation.png','shap_summary.png'])
        image_panel(protocol,year,group,diag)
        selected=metrics[(metrics.protocol==protocol)&(metrics.test_year==year)&(metrics.group==group)]
        if not selected.empty: st.dataframe(selected.T, use_container_width=True)

with tabs[2]:
    st.subheader('Importance across held-out years')
    kind=st.selectbox('Importance measure',['gini_importance','permutation_importance','shap_importance'])
    importance=importance_table(kind)
    part=importance[(importance.protocol==protocol)&(importance.group==group)]
    if part.empty: st.warning('No importance CSVs available.')
    else:
        ranking=part.groupby('feature').importance.mean().sort_values(ascending=False)
        top=st.slider('Number of features',5,min(30,len(ranking)),min(15,len(ranking))) if len(ranking)>=5 else len(ranking)
        chosen=ranking.head(top).index
        grid=part[part.feature.isin(chosen)].pivot_table(index='feature',columns='year',values='importance').reindex(chosen)
        fig=px.imshow(grid,aspect='auto',color_continuous_scale='Viridis',title=f'{protocol} · {group} · {kind.replace("_"," ")}')
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Gini, held-out permutation, and mean absolute SHAP measure different properties; do not compare their raw scales directly. Negative permutation values may occur.')
        st.dataframe(grid,use_container_width=True)

with tabs[3]:
    st.subheader('One-way partial dependence')
    st.caption('The combined PNG contains the one-way PDP panels for a model. These are averaged model predictions, not observed detection rates.')
    if year is not None: image_panel(protocol,year,group,'pdp_1way.png')
    else: st.info('Select a year in the sidebar.')
    with st.expander('Compare all years for this light condition'):
        for yr in sorted(available[available.group==group].test_year.unique()):
            st.markdown(f'**{protocol} — {yr} — {group}**')
            image_panel(protocol,int(yr),group,'pdp_1way.png')

with tabs[4]:
    st.subheader('Two-way PDP contour explorer')
    st.caption('Axes are environmental predictor values; labeled contour levels are mean model-predicted presence probabilities. A contour crossing 0.50 is not a biological threshold.')
    z=archives[protocol]
    pattern=re.compile(rf'^{protocol}/(\d{{4}})_{group}/(pdp2_.*\.png)$')
    entries=[]
    for n in z.namelist():
        m=pattern.match(n)
        if m: entries.append((int(m.group(1)),m.group(2)))
    pairs=sorted(set(fn for _,fn in entries))
    if not pairs: st.warning('No two-way PDP images found.')
    else:
        selected_pair=st.selectbox('Interaction (two environmental variables)',pairs,format_func=lambda s:s.removeprefix('pdp2_').removesuffix('.png').replace('__',' × '))
        if year is not None: image_panel(protocol,year,group,selected_pair)
        st.markdown('#### Compare the same pair across years')
        comparison_years=st.multiselect('Years',sorted(set(y for y,_ in entries)),default=sorted(set(y for y,_ in entries))[-3:])
        for yr in comparison_years:
            if (yr,selected_pair) in entries:
                st.markdown(f'**{protocol} · {yr} · {group}**')
                image_panel(protocol,yr,group,selected_pair)
            else: st.caption(f'{yr}: this interaction was not generated for this model.')

with tabs[5]:
    st.subheader('FORWARD versus LOYO')
    if len(archives)<2: st.info('Upload both ZIPs for comparison.')
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

with tabs[6]:
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
