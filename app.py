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

def check_password():
    if st.session_state.get("authenticated", False):
        return True

    st.title("Shark Research Dashboard")
    password = st.text_input("Enter password", type="password")

    if st.button("Log in"):
        if password == st.secrets["dashboard_password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")

    return False

if not check_password():
    st.stop()


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
    if not list(root.glob('FORWARD_*.zip')) and not list(root.glob('LOYO_*.zip')):
        root = Path(__file__).parent  # Also supports ZIPs uploaded to repository root
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
    year=st.selectbox('Held-out year',sorted(available[available.group==group].test_year.unique().tolist()),index=0)

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
        fig=px.line(m,x='test_year',y=measure,markers=True,title=f'{protocol} · {group}: {measure.upper()} by held-out year')
        fig.update_yaxes(range=[0,1]);st.plotly_chart(fig,use_container_width=True)
        pivot=metrics[metrics.protocol==protocol].pivot(index='group',columns='test_year',values=measure)
        fig=px.imshow(pivot,aspect='auto',color_continuous_scale='Viridis',zmin=0,zmax=1,text_auto='.2f',title=f'{protocol} — all years and groups')
        st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.subheader('Saved model diagnostics')
    if year is None: st.info('Select a year in the sidebar.')
    else:
        diag=st.selectbox('Figure',['confusion_matrix.png','roc.png','gini.png','permutation.png','shap_summary.png'])
        image_panel(protocol,year,group,diag)
        selected=metrics[(metrics.protocol==protocol)&(metrics.test_year==year)&(metrics.group==group)]
        if not selected.empty: st.dataframe(selected.T, use_container_width=True)

with tabs[2]:
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
        with st.expander('Compare this importance measure across years'):
            all_part=importance[(importance.protocol==protocol)&(importance.group==group)]
            chosen=part.head(top).feature.tolist()
            grid=all_part[all_part.feature.isin(chosen)].pivot_table(index='feature',columns='year',values='importance').reindex(chosen)
            st.plotly_chart(px.imshow(grid,aspect='auto',color_continuous_scale='Viridis',title=f'{protocol} · {group} · across years'),use_container_width=True)
    st.caption('Gini, permutation, and mean absolute SHAP measure different properties; their scales are not directly comparable.')

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
    st.subheader(f'Two-way PDP contours — {protocol} · {year} · {group}')
    st.caption('Contours show average model predictions, not observed detection rates or causal effects.')
    z=archives[protocol]
    pattern=re.compile(rf'^{protocol}/{year}_{group}/(pdp2_.*\.png)$')
    pairs=sorted({m.group(1) for n in z.namelist() if (m:=pattern.match(n))})
    if not pairs: st.warning('No two-way PDP images found for this selected model.')
    else:
        selected_pair=st.selectbox('Interaction (two environmental variables)',pairs,format_func=lambda t:t.removeprefix('pdp2_').removesuffix('.png').replace('__',' × '))
        image_panel(protocol,year,group,selected_pair)
        with st.expander('Compare the same interaction across years'):
            pattern_all=re.compile(rf'^{protocol}/(\d{{4}})_{group}/(pdp2_.*\.png)$')
            entries={(int(m.group(1)),m.group(2)) for n in z.namelist() if (m:=pattern_all.match(n))}
            years=sorted({yr for yr,fn in entries if fn==selected_pair})
            comparison_years=st.multiselect('Comparison years',years,default=[])
            for yr in comparison_years: image_panel(protocol,yr,group,selected_pair)

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
