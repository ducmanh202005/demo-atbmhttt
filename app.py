from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from detector import DLPDetector

ROOT = Path(__file__).resolve().parent
FEEDBACK_FILE = ROOT / "data" / "feedback.csv"

st.set_page_config(page_title="ML-based DLP Demo", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_detector() -> DLPDetector:
    return DLPDetector(ROOT / "models")


detector = load_detector()

SAMPLES = {
    "Nội bộ an toàn": {
        "text": "Gửi anh Minh báo cáo họp tuần này của phòng kinh doanh. File chỉ dùng nội bộ và sẽ cập nhật trên hệ thống chung trước 17h.",
        "behavior": {
            "gio_gui": 10,
            "la_ngoai_cong_ty": 0,
            "dung_luong_mb": 3.0,
            "so_lan_gui": 1,
            "ip_la": 0,
            "thiet_bi_la": 0,
            "usb_copy": 0,
            "channel": "Email",
            "sender_department": "Sales",
            "receiver_domain": "internal",
            "file_type": "docx",
            "destination": "internal",
        },
    },
    "Dữ liệu nhạy cảm gửi hợp lệ": {
        "text": "Khách hàng Nguyễn Văn An xác nhận email nguyenvanan@gmail.com và số điện thoại 0912 345 678 để hoàn tất hợp đồng.",
        "behavior": {
            "gio_gui": 14,
            "la_ngoai_cong_ty": 1,
            "dung_luong_mb": 15.0,
            "so_lan_gui": 2,
            "ip_la": 0,
            "thiet_bi_la": 0,
            "usb_copy": 0,
            "channel": "Email",
            "sender_department": "Sales",
            "receiver_domain": "partner.com",
            "file_type": "pdf",
            "destination": "partner",
        },
    },
    "Nguy cơ rò rỉ cao": {
        "text": "Gửi ra gmail cá nhân danh sách khách hàng VIP. Họ tên Nguyễn Minh Quân, CCCD 079304001234, số tài khoản 0123456789012, địa chỉ: số nhà 12 ngõ 80 Nguyễn Trãi, Hà Nội. password: Abc@2024secret",
        "behavior": {
            "gio_gui": 1,
            "la_ngoai_cong_ty": 1,
            "dung_luong_mb": 820.0,
            "so_lan_gui": 11,
            "ip_la": 1,
            "thiet_bi_la": 1,
            "usb_copy": 1,
            "channel": "USB",
            "sender_department": "Finance",
            "receiver_domain": "gmail.com",
            "file_type": "zip",
            "destination": "personal_email",
        },
    },
}

LABELS = {
    "safe": "Bình thường",
    "pii": "Thông tin cá nhân",
    "credential": "Mật khẩu/API key/token",
    "financial": "Tài chính/ngân hàng",
    "confidential": "Tài liệu nội bộ",
    "sensitive": "Nhạy cảm",
    "leak_risk": "Nguy cơ rò rỉ",
}
ACTION_COLOR = {
    "ALLOW": "green",
    "WARN": "orange",
    "ENCRYPT": "blue",
    "QUARANTINE": "darkorange",
    "BLOCK": "red",
}

CHANNELS = ["Email", "Cloud", "USB", "Endpoint", "SharePoint"]
DEPARTMENTS = ["HR", "Finance", "IT", "Sales", "R&D"]
DOMAINS = ["internal", "partner.com", "gmail.com", "unknown", "public_cloud"]
FILE_TYPES = ["txt", "pdf", "docx", "xlsx", "zip", "image", "source"]
DESTINATIONS = ["internal", "partner", "external", "personal_email", "public_cloud"]


def ensure_feedback_file() -> None:
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    if not FEEDBACK_FILE.exists() or FEEDBACK_FILE.stat().st_size == 0:
        FEEDBACK_FILE.write_text(
            "text,doc_label,action,dlp_score,user_feedback\n",
            encoding="utf-8",
        )


def option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


st.title("Demo ML-based DLP")
st.caption(
    "Pipeline: Collection -> PII/Masking -> Text ML -> UEBA/Anomaly ML -> Policy Engine -> Feedback"
)

with st.sidebar:
    st.header("Kịch bản nhanh")
    sample_name = st.selectbox("Chọn mẫu", list(SAMPLES.keys()))
    use_sample = st.button("Nạp kịch bản")

if "form_data" not in st.session_state:
    st.session_state.form_data = SAMPLES["Nội bộ an toàn"]
if use_sample:
    st.session_state.form_data = SAMPLES[sample_name]

tab_analyze, tab_pii, tab_ml, tab_policy, tab_feedback, tab_metrics = st.tabs(
    ["Phân tích DLP", "PII & Masking", "Mô hình ML", "Policy", "Feedback", "Metrics"]
)

with tab_analyze:
    left, right = st.columns([1.55, 1.15])
    with left:
        text = st.text_area(
            "Nội dung email / tài liệu",
            value=st.session_state.form_data["text"],
            height=260,
        )
    with right:
        b = st.session_state.form_data["behavior"]
        channel = st.selectbox("Kênh dữ liệu", CHANNELS, index=option_index(CHANNELS, b["channel"]))
        sender_department = st.selectbox(
            "Phòng ban gửi",
            DEPARTMENTS,
            index=option_index(DEPARTMENTS, b["sender_department"]),
        )
        receiver_domain = st.selectbox(
            "Domain người nhận",
            DOMAINS,
            index=option_index(DOMAINS, b["receiver_domain"]),
        )
        file_type = st.selectbox("Loại file", FILE_TYPES, index=option_index(FILE_TYPES, b["file_type"]))
        destination = st.selectbox(
            "Đích gửi",
            DESTINATIONS,
            index=option_index(DESTINATIONS, b["destination"]),
        )
        gio_gui = st.slider("Giờ gửi", 0, 23, int(b["gio_gui"]))
        dung_luong = st.number_input(
            "Dung lượng file (MB)",
            min_value=0.1,
            max_value=5000.0,
            value=float(b["dung_luong_mb"]),
            step=10.0,
        )
        so_lan = st.number_input("Số lần gửi", min_value=1, max_value=50, value=int(b["so_lan_gui"]))
        la_ngoai = st.checkbox("Người nhận ngoài công ty", value=bool(b["la_ngoai_cong_ty"]))
        ip_la = st.checkbox("IP lạ", value=bool(b["ip_la"]))
        thiet_bi_la = st.checkbox("Thiết bị lạ", value=bool(b["thiet_bi_la"]))
        usb_copy = st.checkbox("Copy ra USB", value=bool(b["usb_copy"]))

    if st.button("Phân tích DLP", type="primary", use_container_width=True):
        behavior = {
            "gio_gui": gio_gui,
            "la_ngoai_cong_ty": int(la_ngoai),
            "dung_luong_mb": dung_luong,
            "so_lan_gui": so_lan,
            "ip_la": int(ip_la),
            "thiet_bi_la": int(thiet_bi_la),
            "usb_copy": int(usb_copy),
            "channel": channel,
            "sender_department": sender_department,
            "receiver_domain": receiver_domain,
            "file_type": file_type,
            "destination": destination,
        }
        st.session_state.last_text = text
        st.session_state.last_behavior = behavior
        st.session_state.last_result = detector.analyze(text, behavior)

    result = st.session_state.get("last_result")
    if result:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PII phát hiện", sum(len(v) for v in result["pii"].values()))
        c2.metric("Nhãn tài liệu", LABELS.get(result["doc_label"], result["doc_label"]))
        c3.metric("Behavior RF", f"{result['behavior_probability'] * 100:.1f}%")
        c4.metric("Anomaly risk", f"{result['anomaly_risk'] * 100:.1f}%")
        c5.metric("Quyết định", result["action"])

        st.markdown(
            f"<h2 style='color:{ACTION_COLOR[result['action']]}'>{result['action']}</h2>",
            unsafe_allow_html=True,
        )
        st.write(f"**DLP score:** {result['dlp_score'] * 100:.1f}/100")
        st.info(result["reason"])

with tab_pii:
    result = st.session_state.get("last_result")
    if not result:
        st.info("Hãy chạy phân tích ở tab đầu tiên trước.")
    else:
        c1, c2 = st.columns([1, 1.25])
        with c1:
            st.subheader("PII phát hiện")
            if result["pii"]:
                for name, values in result["pii"].items():
                    st.write(f"**{name}**: {', '.join(values)}")
            else:
                st.success("Không phát hiện PII rõ ràng.")
        with c2:
            st.subheader("Văn bản sau khi ẩn danh")
            st.code(result["masked_text"])

with tab_ml:
    result = st.session_state.get("last_result")
    if not result:
        st.info("Hãy chạy phân tích ở tab đầu tiên trước.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Text Classification")
            st.write(f"**Model đang dùng:** {result['text_model_name']}")
            st.write(f"**Nhãn dự đoán:** {LABELS.get(result['doc_label'], result['doc_label'])}")
            st.write(f"**Độ tin cậy:** {result['doc_confidence'] * 100:.1f}%")
            st.json({k: round(v, 4) for k, v in result["doc_probabilities"].items()})
        with c2:
            st.subheader("UEBA / Anomaly Detection")
            st.progress(
                min(max(result["behavior_probability"], 0.0), 1.0),
                text=f"Random Forest risk = {result['behavior_probability'] * 100:.1f}%",
            )
            st.progress(
                min(max(result["anomaly_risk"], 0.0), 1.0),
                text=f"Anomaly ensemble risk = {result['anomaly_risk'] * 100:.1f}%",
            )
            st.json({k: round(v, 4) for k, v in result["anomaly_scores"].items()})

        st.subheader("Feature importance nổi bật")
        top_feats = sorted(result["feature_importances"].items(), key=lambda x: x[1], reverse=True)[:10]
        st.dataframe(pd.DataFrame(top_feats, columns=["feature", "importance"]), use_container_width=True)

with tab_policy:
    result = st.session_state.get("last_result")
    if not result:
        st.info("Hãy chạy phân tích ở tab đầu tiên trước.")
    else:
        st.subheader("Ngưỡng quyết định")
        st.table(
            pd.DataFrame(
                [
                    ["< 0.28", "ALLOW", "Cho phép"],
                    ["0.28 - 0.42", "WARN", "Cảnh báo và ghi log"],
                    ["0.42 - 0.58", "ENCRYPT", "Yêu cầu mã hóa/kênh an toàn"],
                    ["0.58 - 0.78", "QUARANTINE", "Cách ly để kiểm tra"],
                    [">= 0.78", "BLOCK", "Chặn hoàn toàn"],
                ],
                columns=["DLP score", "Action", "Ý nghĩa"],
            )
        )
        st.write(f"**Quyết định hiện tại:** {result['action']}")
        st.write(f"**Score:** {result['dlp_score']:.4f}")
        st.write(result["reason"])

with tab_feedback:
    result = st.session_state.get("last_result")
    if not result:
        st.info("Hãy chạy phân tích ở tab đầu tiên trước.")
    else:
        ensure_feedback_file()
        feedback = st.selectbox(
            "Phản hồi của admin",
            ["true_positive", "false_positive", "false_negative", "valid_business_case"],
        )
        if st.button("Lưu phản hồi"):
            row = pd.DataFrame(
                [
                    {
                        "text": st.session_state.last_text,
                        "doc_label": result["doc_label"],
                        "action": result["action"],
                        "dlp_score": round(float(result["dlp_score"]), 4),
                        "user_feedback": feedback,
                    }
                ]
            )
            row.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False, encoding="utf-8")
            st.success("Đã lưu phản hồi vào data/feedback.csv")

        if FEEDBACK_FILE.exists():
            st.subheader("Feedback đã lưu")
            st.dataframe(pd.read_csv(FEEDBACK_FILE), use_container_width=True)

with tab_metrics:
    st.subheader("Chỉ số mô hình đã train")
    st.json(detector.metrics)

st.markdown("---")
st.caption("Train lại bằng: python train_models.py. App chỉ load model đã train sẵn từ thư mục models/.")
