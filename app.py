from __future__ import annotations

from pathlib import Path

import streamlit as st

from detector import DLPDetector

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='DLP Demo Final', page_icon='🛡️', layout='wide')

@st.cache_resource
def load_detector() -> DLPDetector:
    return DLPDetector(ROOT / 'models')


detector = load_detector()

SAMPLES = {
    'Nội bộ an toàn': {
        'text': 'Gửi anh Minh báo cáo họp tuần này của phòng kinh doanh. File chỉ dùng nội bộ và sẽ cập nhật trên hệ thống chung trước 17h.',
        'behavior': {'gio_gui': 10, 'la_ngoai_cong_ty': 0, 'dung_luong_mb': 3.0, 'so_lan_gui': 1, 'ip_la': 0, 'thiet_bi_la': 0, 'usb_copy': 0},
    },
    'Có dữ liệu nhạy cảm': {
        'text': 'Khách hàng Nguyễn Văn An xác nhận email nguyenvanan@gmail.com và số điện thoại 0912 345 678 để hoàn tất hợp đồng.',
        'behavior': {'gio_gui': 14, 'la_ngoai_cong_ty': 0, 'dung_luong_mb': 15.0, 'so_lan_gui': 2, 'ip_la': 0, 'thiet_bi_la': 0, 'usb_copy': 0},
    },
    'Nguy cơ rò rỉ cao': {
        'text': 'Gửi ra gmail cá nhân danh sách khách hàng VIP. Họ tên Nguyễn Minh Quân, CCCD 079304001234, số tài khoản 0123456789012, địa chỉ: số nhà 12 ngõ 80 Nguyễn Trãi, Hà Nội. password: Abc@2024secret',
        'behavior': {'gio_gui': 1, 'la_ngoai_cong_ty': 1, 'dung_luong_mb': 820.0, 'so_lan_gui': 11, 'ip_la': 1, 'thiet_bi_la': 1, 'usb_copy': 1},
    },
}

LABELS = {'safe': 'An toàn', 'sensitive': 'Nhạy cảm', 'leak_risk': 'Nguy cơ rò rỉ'}
ACTION_COLOR = {'ALLOW': 'green', 'WARN': 'orange', 'QUARANTINE': 'darkorange', 'BLOCK': 'red'}

st.title('Demo DLP có train model')
st.caption('Pipeline: PII detection -> Text classification (TF-IDF + Logistic Regression) -> Behavior analysis (Random Forest) -> Policy decision')

with st.sidebar:
    st.header('Kịch bản nhanh')
    sample_name = st.selectbox('Chọn mẫu', list(SAMPLES.keys()))
    use_sample = st.button('Nạp kịch bản')

if 'form_data' not in st.session_state:
    st.session_state.form_data = SAMPLES['Nội bộ an toàn']
if use_sample:
    st.session_state.form_data = SAMPLES[sample_name]

left, right = st.columns([1.7, 1.1])
with left:
    text = st.text_area('Nội dung email / tài liệu', value=st.session_state.form_data['text'], height=220)
with right:
    b = st.session_state.form_data['behavior']
    gio_gui = st.slider('Giờ gửi', 0, 23, int(b['gio_gui']))
    la_ngoai = st.checkbox('Người nhận ngoài công ty', value=bool(b['la_ngoai_cong_ty']))
    dung_luong = st.number_input('Dung lượng file (MB)', min_value=0.1, max_value=5000.0, value=float(b['dung_luong_mb']), step=10.0)
    so_lan = st.number_input('Số lần gửi', min_value=1, max_value=50, value=int(b['so_lan_gui']))
    ip_la = st.checkbox('IP lạ', value=bool(b['ip_la']))
    thiet_bi_la = st.checkbox('Thiết bị lạ', value=bool(b['thiet_bi_la']))
    usb_copy = st.checkbox('Copy ra USB', value=bool(b['usb_copy']))

if st.button('Phân tích DLP', type='primary', use_container_width=True):
    behavior = {
        'gio_gui': gio_gui,
        'la_ngoai_cong_ty': int(la_ngoai),
        'dung_luong_mb': dung_luong,
        'so_lan_gui': so_lan,
        'ip_la': int(ip_la),
        'thiet_bi_la': int(thiet_bi_la),
        'usb_copy': int(usb_copy),
    }
    result = detector.analyze(text, behavior)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('PII loại phát hiện', sum(len(v) for v in result['pii'].values()))
    c2.metric('Nhãn tài liệu', LABELS.get(result['doc_label'], result['doc_label']))
    c3.metric('Xác suất hành vi bất thường', f"{result['behavior_probability']*100:.1f}%")
    c4.metric('Quyết định', result['action'])

    st.markdown('---')
    r1, r2 = st.columns([1.2, 1.0])
    with r1:
        st.subheader('1) Phát hiện thông tin nhạy cảm (PII)')
        if result['pii']:
            for k, vals in result['pii'].items():
                st.write(f"**{k}**: {', '.join(vals)}")
        else:
            st.success('Không phát hiện PII rõ ràng.')

        st.subheader('2) Phân loại tài liệu')
        st.write(f"**Nhãn dự đoán:** {LABELS.get(result['doc_label'], result['doc_label'])}")
        st.write(f"**Độ tin cậy:** {result['doc_confidence']*100:.1f}%")
        st.json({k: round(v, 4) for k, v in result['doc_probabilities'].items()})

    with r2:
        st.subheader('3) Phân tích hành vi')
        st.progress(min(max(result['behavior_probability'], 0.0), 1.0), text=f"Risk = {result['behavior_probability']*100:.1f}%")
        top_feats = sorted(result['feature_importances'].items(), key=lambda x: x[1], reverse=True)[:5]
        st.write('**Yếu tố mô hình dùng nhiều:**')
        for name, score in top_feats:
            st.write(f'- {name}: {score:.3f}')

        st.subheader('4) Chính sách phản hồi')
        st.markdown(f"<h3 style='color:{ACTION_COLOR[result['action']]}'>{result['action']}</h3>", unsafe_allow_html=True)
        st.write(f"**DLP score:** {result['dlp_score']*100:.1f}/100")
        st.info(result['reason'])

with st.expander('Xem chỉ số mô hình đã train'):
    st.json(detector.metrics)

st.markdown('---')
st.caption('Chạy train riêng bằng: python train_models.py. App chỉ load model đã train sẵn từ thư mục models/.')
