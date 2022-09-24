import os
import json
import streamlit as st
from matplotlib import pyplot as plt

def load_data(filename):
    
    filename = os.path.join(DATA_DIR, filename)
    
    if "filename" not in st.session_state:
        st.session_state.filename = filename
        with open(st.session_state.filename, "r") as f:
            st.session_state.pseudo_labels = json.load(f)
    
    if filename != st.session_state.filename:         
        st.session_state.filename = filename
        with open(st.session_state.filename, "r") as f:
            st.session_state.pseudo_labels = json.load(f)
        
def reset(label):
    
    if "idx" not in st.session_state:
        st.session_state.idx = 0
        st.session_state.label = label
        st.session_state.max_index = len(st.session_state.pseudo_labels[label])
    if label != st.session_state.label:
        st.session_state.label = label
        st.session_state.max_index = len(st.session_state.pseudo_labels[label])
        st.session_state.idx = 0

def save_analysis():
    with open(st.session_state.filename, "w") as f:
        json.dump(st.session_state.pseudo_labels, f, indent=4)
    st.balloons()

def set_index(idx):
    try:
        idx = int(idx)
    except:
        return 0
    
    if idx < st.session_state.max_index and idx > 0:
        st.session_state.idx = idx
        
def next_image():
    if st.session_state.idx < st.session_state.max_index - 1:
        st.session_state.idx += 1
    else:
        st.session_state.idx = 0

def previous_image():
    if st.session_state.idx > 0:
        st.session_state.idx -= 1
    else:
        st.session_state.idx = st.session_state.max_index-1  

             
st.set_page_config(layout="wide")
st.title("Pseudo Labels Analysis", anchor=None)

DATA_DIR = "data"

file_col, _ = st.columns([4, 10])
with file_col:
    filename = st.selectbox(
        label='Which file to analyze?',
        options=os.listdir(DATA_DIR)
    )
if filename is not None:
    load_data(filename)
    st.header("Annotation")
    label_col, progress_txt, _ = st.columns([4, 4, 10])
    with label_col:
        label = st.selectbox(
            label="Which label you want to fix?",
            options=(st.session_state.pseudo_labels.keys())
        )
    if label is not None:
        # reset i
        reset(label=label)
        progress_txt.text_input(label="Progress", value=f"{st.session_state.idx}/{st.session_state.max_index}", disabled=True)
        
        annotation_container = st.container()
        with annotation_container:
            st.write(f"For label **{st.session_state.label}** you have **{st.session_state.max_index}** images to check.")
            
            prev_btn, next_btn, _, output_col, _ = st.columns([2, 2, 2, 2, 10])
            prev_btn.button("Previous Image", on_click=previous_image)
            next_btn.button("Next Image", on_click=next_image)
            output_col.button(
                label="Save Analysis",
                on_click=save_analysis
            )
            col_img, _, col_ann, _,  col_stats = st.columns([3, 1, 3, 1, 5])
            with col_img:
                st.header("Image")
                file_path = st.session_state.pseudo_labels[label][st.session_state.idx]["file_path"]
                gt_label = st.session_state.pseudo_labels[label][st.session_state.idx]["ground_truth"]
                st.write(f"Ground Truth Label: **{gt_label}**")
                st.image(file_path)
                
            with col_ann:
                st.header("Actions")
                pred_label = st.session_state.pseudo_labels[label][st.session_state.idx]["prediction"]
                pred_score = st.session_state.pseudo_labels[label][st.session_state.idx]["score"]
                st.write(f"Suggested new label is **{pred_label}** with score **{pred_score:.4f}**")
                new_label = st.radio(
                    label="Choose the new label",
                    options=list(st.session_state.pseudo_labels.keys()) + ["unknown"],
                )
                if "new_label" in st.session_state.pseudo_labels[label][st.session_state.idx].keys():
                    nl=st.session_state.pseudo_labels[label][st.session_state.idx]['new_label']
                    if nl!=gt_label:           
                        st.write(f"You already changed the label for this image from **{gt_label}** to **{nl}**")
                if st.button("Save Change"):
                    st.session_state.pseudo_labels[label][st.session_state.idx]["new_label"] = new_label
            
            with col_stats:
                st.header("Stats")
                
                plot_data = {}
                unknowns = 0
                for l in st.session_state.pseudo_labels:
                    changes = 0
                    for pseudo_label in st.session_state.pseudo_labels[l]:
                        if "new_label" in pseudo_label:
                            if pseudo_label["new_label"] != pseudo_label["ground_truth"]:
                                if pseudo_label["new_label"] == "unknown":
                                    unknowns += 1
                                else:
                                    changes += 1
                    plot_data[l] = 100*changes / len(st.session_state.pseudo_labels[l])
                
                
                plot_data['unknown'] = 100*unknowns / sum([len(st.session_state.pseudo_labels[l]) for l in st.session_state.pseudo_labels])
                for l in plot_data:
                    st.markdown(f"* Change rate in {l} images is {plot_data[l]:.4f}%")
                fig, ax = plt.subplots()
                names = list(plot_data.keys())
                values = list(plot_data.values())
                ax.bar(range(len(plot_data)), values, tick_label=names)
                ax.set_ylim(bottom=0, top=50)
                ax.set_ylabel(f"% Wrong Annotations")
                ax.set_xlabel("Label")
                col_stats.pyplot(fig)


    