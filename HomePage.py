import streamlit as st
#import lib.common as tools

st.set_page_config(
    page_title="Đồ án cuối kỳ",
    page_icon="☕",
)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-042.jpg");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)


# logo_path = "./VCT.png"
# st.sidebar.image(logo_path, width=200)

st.write("# Đồ án cuối kỳ")
st.write("# 20133044 - Lê Huy Hoàng")
st.write('# 20142481 - Đàm Trọng Hải Dương')
st.write("# Mã lớp : DIPR430685_23_1_02")

st.markdown(
    """
    ## Sản phẩm
    Project cuối kỳ cho môn học xử lý ảnh số DIPR430685.
    Thuộc Trường Đại Học Sư Phạm Kỹ Thuật TP.HCM.
    ### 5 chức năng chính trong bài
    - 📖Nhận diện khuôn mặt
    - 📖Phát hiện khuôn mặt
    - 📖Nhận dạng đối tượng yolo4
    - 📖Nhận dạng 5 loại xe 
    - 📖Xử lý ảnh số
    ## Thông tin sinh viên thực hiện
    - 🧑Họ tên: Lê Huy Hoàng
    - 💳MSSV: 20133044
    - 🧑Họ tên: Đàm Trọng Hải Dương
    - 💳MSSV: 20142481
    """
)


