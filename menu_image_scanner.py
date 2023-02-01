import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import vision    
import io  
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter  
import fitz
import base64
from pathlib import Path
import copy

## Google cloud vision credentials

credentials = service_account.Credentials.from_service_account_file(**st.secrets.gc_api)

client = vision.ImageAnnotatorClient(credentials=credentials)
# client = vision.ImageAnnotatorClient(credentials=**st.secrets.gc_api)

## Image to text functions

def image_processing(image):
    
    img = im.open(image)
    
    # convert to binary
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    delta = 1
    limit = 30
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    
    img_post = img.rotate(best_angle)
    
    img_post.save('processed_image.png')

def extract_text_img(image_path):
    
    ## Extract text from image
    with io.open(image_path, 'rb') as image_file:        
        content = image_file.read()    
    image = vision.Image(content=content)    
    
    response = client.text_detection(image=image)    
    texts = response.text_annotations  
    
    return response, texts

def extract_menu_items_img(response, texts):

    ## For better grouping of words - use grouped words extracted by google cloud vision
    ## Get location of each group of words
    i = 0
    menu_items = pd.DataFrame()
    menu_list = texts[0].description.split('\n')
    full_word = ''

    for text in [text.description for text in texts[1:]]:
        if (text in menu_list[i]) & (full_word.replace(' ','') != menu_list[i].replace(' ','')):
            menu_items = pd.concat([menu_items, pd.DataFrame({'item':[text], 'text':[menu_list[i]], 'group': [i]})])

            if full_word == '':
                full_word = full_word + text
            else:
                full_word = full_word + ' ' + text

        else:
            i += 1
            menu_items = pd.concat([menu_items, pd.DataFrame({'item':[text], 'text':[menu_list[i]], 'group': [i]})])
            full_word = text

    menu_items = menu_items.reset_index(drop=True)
    menu_items['is_price'] = menu_items['item'].str.lower().str.replace('p','').str.replace('.','').str.replace(' ','').str.replace('$','').str.replace('%','').str.replace('-','').str.isdigit()    
    menu_items['group2'] = ((menu_items['group']!=menu_items['group'].shift(1))|(menu_items['is_price'])).cumsum().fillna(0) 
    menu_items = menu_items.merge(menu_items.groupby(['group2'])[['item']].agg(' '.join).reset_index().rename(columns={'item':'item2'}), on='group2', how='left')    
    
    ## Get the coordinates of each individual text on the image
    menu_df1 = pd.DataFrame()

    for i in range(4):
        menu_df1['text'] = [text.description for text in texts[1:]]
        menu_df1[f'x{i+1}'] = [vertex[i].x for vertex in (text.bounding_poly.vertices for text in texts[1:])]
        menu_df1[f'y{i+1}'] = [vertex[i].y for vertex in (text.bounding_poly.vertices for text in texts[1:])]
    
    menu_items = pd.concat([menu_items, menu_df1.iloc[:,1:]], axis=1)
    
    ## Get the coordinates of the group of texts on the image
    menu_df = menu_items.groupby(['item2','group2']).agg({'x1':'min','y1':'min','x2':'max','y2':'min','x3':'max','y3':'max','x4':'min','y4':'max'}).reset_index()
    menu_df = menu_df.rename(columns={'item2':'text','group2':'group'})
    
    menu_df['x_min'] = menu_df[['x1','x2','x3','x4']].min(axis=1)
    menu_df['x_max'] = menu_df[['x1','x2','x3','x4']].max(axis=1)
    menu_df['y_min'] = menu_df[['y1','y2','y3','y4']].min(axis=1)
    menu_df['y_max'] = menu_df[['y1','y2','y3','y4']].max(axis=1)
    menu_df['x_mid'] = menu_df[['x_min','x_max']].mean(axis=1)
    menu_df['y_mid'] = menu_df[['y_min','y_max']].mean(axis=1)

    ## Identify words that are on the same row to match menu item with price
    menu_df = menu_df.sort_values(by=['y_min','x_min']).reset_index(drop=True)
    
    menu_df['is_price'] = menu_df['text'].str.lower().str.replace('p','').str.replace('.','').str.replace(' ','').str.replace('$','').str.replace('%','').str.replace('-','').str.isdigit()
    menu_df['is_same_row'] = (~((menu_df['y_mid'].between(menu_df['y_min'].shift(-1), menu_df['y_max'].shift(-1)))|(
                            menu_df['y_mid'].shift(-1).between(menu_df['y_min'], menu_df['y_max']))))
    
    menu_df['group2'] = menu_df['is_same_row'].shift(1).cumsum().fillna(0)

    menu_df = menu_df.sort_values(by=['group2','x_min']).reset_index(drop=True)
    menu_df['is_same_row2'] = (menu_df['group2']!=menu_df['group2'].shift(-1))|((menu_df['is_price']==False)&(menu_df['is_price'].shift(-1) == True)).shift(-1)
    menu_df['group3'] = menu_df['is_same_row2'].shift(1).cumsum().fillna(0)

    menu_final = menu_df[menu_df['is_price']==False].groupby(['group3'], sort=False)[['text']].agg(' '.join).reset_index()
    menu_final = menu_final.merge(menu_df[menu_df['is_price']==True].groupby(['group3'])[['text']].agg('/'.join).reset_index(), on='group3', how='left')
    menu_final.columns = ['group','item','price']
    menu_final['item'] = menu_final['item'].str.strip('.-,/・')
    menu_final['price'] = menu_final['price'].str.strip('.-,/・')
    
    return menu_df, pd.DataFrame({'item':menu_list}), menu_final[['item','price']]


## PDF to text functions

def extract_text_pdf(pdf_path):

    #file_path.read()).decode('utf-8')
    #doc = fitz.open(base64.b64encode(pdf_path.read()).decode('utf-8'))
    #"data:application/pdf;base64,{base64_pdf}"
    #base64_pdf = base64.b64encode(pdf_path.read()).decode('utf-8')
    #doc = fitz.open(base64_pdf)
    
    doc = fitz.open(stream=pdf_path.read(), filetype="pdf")    
    #doc = fitz.open(pdf_path)  
    return doc

def extract_menu_items_pdf(doc):

    menu_all = pd.DataFrame()
    menu_final = pd.DataFrame()
    
    for p in range(0, doc.page_count):
        page = doc.load_page(p)
        
        ## Tag item prices
        i = 0
        menu_items = pd.DataFrame()
        menu_list = [text.strip() for text in page.get_text().split('\n')]
        while("" in menu_list):
            menu_list.remove("")
        full_word = ''

        for text in [text[4] for text in page.get_text_words()]:
            if (text in menu_list[i]) & (full_word.replace(' ','') != menu_list[i].replace(' ','')):
                menu_items = pd.concat([menu_items, pd.DataFrame({'item':[text], 'text':[menu_list[i]], 'group': [i]})])

                if full_word == '':
                    full_word = full_word + text
                else:
                    full_word = full_word + ' ' + text

            else:
                i += 1
                menu_items = pd.concat([menu_items, pd.DataFrame({'item':[text], 'text':[menu_list[i]], 'group': [i]})])
                full_word = text

        menu_items = menu_items.reset_index(drop=True)
        menu_items['is_price'] = menu_items['item'].str.lower().str.replace('p','').str.replace('.','').str.replace(' ','').str.replace('$','').str.replace('%','').str.replace('-','').str.replace(',','').str.isdigit()    
        menu_items['group2'] = ((menu_items['group']!=menu_items['group'].shift(1))|((menu_items['is_price'])&(menu_items['group']!=menu_items['group'].shift(-1)))).cumsum().fillna(0) 
        menu_items = menu_items.merge(menu_items.groupby(['group2'])[['item']].agg(' '.join).reset_index().rename(columns={'item':'item2'}), on='group2', how='left') 
        menu_items['is_price2'] = menu_items['item2'].str.lower().str.replace('p','').str.replace('.','').str.replace(' ','').str.replace('$','').str.replace('%','').str.replace('-','').str.replace(',','').str.isdigit()    
        
        menu_final_item = menu_items[menu_items['is_price2']==False][['group2','item2']].drop_duplicates().reset_index(drop=True)
        menu_final_item.columns = ['group2','item']
        menu_final_item['page'] = p+1

        menu_final_price = menu_items[menu_items['is_price2']==True][['group2','item2']].drop_duplicates().reset_index(drop=True)
        menu_final_price['group2'] = menu_final_price['group2'] - 1
        menu_final_price.columns = ['group2','price']

        menu_final = pd.concat([menu_final, menu_final_item.merge(menu_final_price, on='group2', how='outer')]).reset_index(drop=True)
        
        menu_all_page = pd.DataFrame({'item':menu_list})
        menu_all_page['page'] = p+1
        menu_all = pd.concat([menu_all, menu_all_page]).reset_index(drop=True)
      
    return menu_all[['page','item']], menu_final[['page','item','price']]  


## Convert file to text

def convert_to_text(file):
    if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
    
        image_processing(image = file)
        response, texts = extract_text_img(image_path = 'processed_image.png')
        menu_df, menu_all, menu_clean = extract_menu_items_img(response = response, texts = texts)
        
    if file.name.lower().endswith(('.pdf')):
    
        doc = extract_text_pdf(pdf_path = file) 
        test = extract_menu_items_pdf(doc = doc)
        menu_all, menu_clean = extract_menu_items_pdf(doc = doc)
    
    return menu_all, menu_clean

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def show_pdf(file_path):
    base64_pdf = base64.b64encode(file_path.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="400" height="250" type="application/pdf">' 
    st.markdown(pdf_display, unsafe_allow_html=True)


## Streamlit interface

st.title("Menu Image Scanner")
st.markdown("This web app converts menu images or pdf to text.")
st.subheader("")
st.subheader("Input vendor id and name here:")

col1, col2 = st.columns([1,3])
with col1:
    vendor_id = st.text_input(label='Vendor ID')
with col2:
    vendor_name = st.text_input(label='Vendor Name')

st.subheader("")
st.subheader("Select whether you want to upload an image / pdf or take a photo.")
tab1, tab2 = st.tabs(["Upload Image or PDF", "Take a Photo"])


with tab1:
    if 'result1' not in st.session_state:
        st.session_state.result1 = None
        st.session_state.result2 = None

    menu_raw_all = pd.DataFrame()
    menu_clean_all = pd.DataFrame()
    uploaded_files = st.file_uploader(label="Upload Image or PDF file here:", type=['png','jpg','jpeg','pdf'], accept_multiple_files=True)
    uploaded_files2 = copy.deepcopy(uploaded_files)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files2:
            menu_raw, menu_clean = convert_to_text(file=uploaded_file)
            menu_raw.insert(0,'source',uploaded_file.name)
            menu_clean.insert(0,'source',uploaded_file.name)
            menu_raw_all = pd.concat([menu_raw_all, menu_raw])
            menu_clean_all = pd.concat([menu_clean_all, menu_clean])
            menu_clean_all = menu_clean_all.fillna("-")
            st.session_state.result1 = menu_raw_all
            st.session_state.result2 = menu_clean_all
		
        try:
            for i in range(0, len(uploaded_files)//5+1):
                cols = st.columns(5)
                if (len(uploaded_files) - 5*(i+1)) < 0:
                    for j in range(0, len(uploaded_files) - 5*i):
                        if uploaded_files[5*i+j].name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            cols[j].image(uploaded_files[5*i+j], width=100, caption=uploaded_files[5*i+j].name)
                        elif uploaded_files[5*i+j].name.lower().endswith(('.pdf')):
                            show_pdf(uploaded_files[5*i+j])
                            #fitz.open(stream=uploaded_files[5*i+j].read(), filetype="pdf") 
                else: 
                    for j in range(0,5):
                        if uploaded_files[5*i+j].name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            cols[j].image(uploaded_files[5*i+j], width=100, caption=uploaded_files[5*i+j].name)
                        elif uploaded_files[5*i+j].name.lower().endswith(('.pdf')):
                            show_pdf(uploaded_files[5*i+j])
	    
            st.header("")
            st.subheader("Convert Image / PDF to text:")

            if st.button("Convert to Text", key=1):
                st.header("")
                for uploaded_file in uploaded_files2:
                    try:
                        menu_raw, menu_clean = convert_to_text(file=uploaded_file)
                        menu_raw.insert(0,'source',uploaded_file.name)
                        menu_clean.insert(0,'source',uploaded_file.name)
                        menu_raw_all = pd.concat([menu_raw_all, menu_raw])
                        menu_clean_all = pd.concat([menu_clean_all, menu_clean])
                        menu_clean_all = menu_clean_all.fillna("-")
                        st.session_state.result1 = menu_raw_all
                        st.session_state.result2 = menu_clean_all

                    except:
                        st.error(body=f"There seems to be an error in reading the file {uploaded_file.name}. Please re-upload the file.")

        except:
            st.error(body='There seems to be an error in reading the uploaded file/s. Please re-upload the file/s.')


    if (st.session_state.result1 is not None):
        if (len(st.session_state.result1) > 0) & (set([x.name for x in uploaded_files])==set(st.session_state.result1['source'])):

            st.subheader("Clean Menu Items:")
            st.download_button("Download", convert_df(st.session_state.result2),f"{vendor_id}_{vendor_name}_menu_clean.csv","text/csv", key=2)
            st.dataframe(data=st.session_state.result2.reset_index(drop=True), width=1000, height=None)

            st.subheader("Raw Menu Items:")
            st.download_button("Download", convert_df(st.session_state.result1),f"{vendor_id}_{vendor_name}_menu_raw.csv","text/csv", key=3)    
            st.dataframe(data=st.session_state.result1.reset_index(drop=True), width=1000, height=None)

        else:
            st.session_state.result1 = None
            st.session_state.result1 = None


with tab2:
    if 'result3' not in st.session_state:
        st.session_state.result3 = None
        st.session_state.result4 = None

    menu_raw_all = pd.DataFrame()
    menu_clean_all = pd.DataFrame()
    uploaded_image = st.camera_input(label="Take a photo of the menu here:")

    st.header("")
    st.subheader("Convert Image to text:")
    if st.button("Convert to Text",key=4):
        if uploaded_image is not None:
            try:
                #st.image(uploaded_image)
                menu_raw, menu_clean = convert_to_text(file=uploaded_image)
                menu_raw_all = pd.concat([menu_raw_all, menu_raw])
                menu_clean_all = pd.concat([menu_clean_all, menu_clean])
                menu_clean_all = menu_clean_all.fillna("-")
                st.session_state.result3 = menu_raw_all
                st.session_state.result4 = menu_clean_all
            except:
                st.error(body='There seems to be an error in reading the captured image. Please retake the photo.')

    if (st.session_state.result3 is not None):
        if (len(st.session_state.result3) > 0) & (uploaded_image is not None):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Clean Menu Items:")
                st.download_button("Download", convert_df(st.session_state.result4),f"{vendor_id}_{vendor_name}_menu_clean.csv","text/csv")
                st.dataframe(data=st.session_state.result4.reset_index(drop=True), width=500, height=None)
            with col2:
                st.subheader("Raw Menu Items:")
                st.download_button("Download", convert_df(st.session_state.result3),f"{vendor_id}_{vendor_name}_menu_raw.csv","text/csv")    
                st.dataframe(data=st.session_state.result3.reset_index(drop=True), width=500, height=None)

        else:
            st.session_state.result3 = None
            st.session_state.result4 = None
