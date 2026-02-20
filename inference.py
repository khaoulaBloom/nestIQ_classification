import os
import streamlit as st
import torch

from model import MultiTaskPropertyModel


MODEL_PATH = "saved_model.pt"


def _market_demand_probability(features: torch.Tensor) -> float:
    area, bedrooms, bathrooms, distance, age = features[0].tolist()

    area_n = max(0.0, min(area / 500.0, 1.0))
    bed_n = max(0.0, min(bedrooms / 6.0, 1.0))
    bath_n = max(0.0, min(bathrooms / 4.0, 1.0))
    dist_n = max(0.0, min(1.0 - (distance / 30.0), 1.0))
    age_n = max(0.0, min(1.0 - (age / 80.0), 1.0))

    market_score = (
        0.32 * area_n
        + 0.24 * bed_n
        + 0.20 * bath_n
        + 0.16 * dist_n
        + 0.12 * age_n
        - 0.40
    )
    market_prob = torch.sigmoid(torch.tensor(5.0 * market_score)).item()
    return float(market_prob)


def _heuristic_prediction(features: torch.Tensor) -> tuple[float, float, float]:
    area, bedrooms, bathrooms, distance, age = features[0].tolist()

    sale_price = (
        area * 2800
        + bedrooms * 38000
        + bathrooms * 29000
        - distance * 6200
        - age * 1700
    )
    sale_price = max(sale_price, 85000)

    rent_price = (
        area * 8.8
        + bedrooms * 190
        + bathrooms * 170
        - distance * 45
        - age * 7
    )
    rent_price = max(rent_price, 500)

    demand_prob = _market_demand_probability(features)
    return float(sale_price), float(rent_price), float(demand_prob)


def _calibrate_demand_probability(features: torch.Tensor, base_prob: float) -> float:
    """Apply domain calibration so extreme low-attractiveness properties map to low demand."""
    area, bedrooms, bathrooms, distance, age = features[0].tolist()
    demand_prob = float(base_prob)

    # Hard low-demand rule requested: very small and very old homes should be low demand.
    if area <= 55 and age >= 50:
        demand_prob = min(demand_prob, 0.25)
    elif area <= 70 and age >= 45:
        demand_prob = min(demand_prob, 0.38)

    # Soft adjustments for feature quality.
    if bedrooms <= 1:
        demand_prob -= 0.07
    if bathrooms <= 1:
        demand_prob -= 0.06
    if distance >= 22:
        demand_prob -= 0.08
    if area >= 220 and age <= 20:
        demand_prob += 0.06

    return max(0.0, min(1.0, demand_prob))


@st.cache_resource
def load_artifacts() -> tuple[MultiTaskPropertyModel, torch.Tensor, torch.Tensor, bool]:
    model = MultiTaskPropertyModel(input_dim=5, hidden_dim=64)
    reg_mean = torch.tensor([400000.0, 2400.0], dtype=torch.float32)
    reg_std = torch.tensor([160000.0, 900.0], dtype=torch.float32)

    if not os.path.exists(MODEL_PATH):
        return model.eval(), reg_mean, reg_std, False

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        reg_mean = checkpoint.get("reg_mean", reg_mean).float()
        reg_std = checkpoint.get("reg_std", reg_std).float()
        model.eval()
        return model, reg_mean, reg_std, True

    return model.eval(), reg_mean, reg_std, False


def predict_property(features: torch.Tensor) -> tuple[float, float, str, str]:
    model, reg_mean, reg_std, has_trained = load_artifacts()

    if not has_trained:
        sale_price, rent_price, demand_prob = _heuristic_prediction(features)
    else:
        with torch.no_grad():
            reg_pred_norm, cls_logit = model(features)
            reg_pred = reg_pred_norm * reg_std.unsqueeze(0) + reg_mean.unsqueeze(0)
            sale_price = max(float(reg_pred[0, 0].item()), 85000.0)
            rent_price = max(float(reg_pred[0, 1].item()), 500.0)
            demand_prob = float(torch.sigmoid(cls_logit).item())

    demand_prob = _calibrate_demand_probability(features, demand_prob)

    if demand_prob >= 0.7:
        demand_label = "High Demand"
    elif demand_prob <= 0.4:
        demand_label = "Low Demand"
    else:
        demand_label = "Moderate Demand"

    demand_probability = f"{demand_prob * 100:.0f}%"
    return sale_price, rent_price, demand_label, demand_probability


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        [data-testid="stHeader"], #MainMenu, footer { visibility: hidden; height: 0; }

        :root {
            --txt: #3a4674;
            --muted: #63739b;
            --shell-border: rgba(255, 255, 255, 0.52);
            --card-border: rgba(255, 255, 255, 0.42);
        }

        .stApp {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background:
                radial-gradient(circle at 9% 17%, rgba(172, 202, 239, .62), transparent 36%),
                radial-gradient(circle at 88% 11%, rgba(223, 202, 237, .50), transparent 34%),
                linear-gradient(145deg, #c6d7ee 0%, #dde4f3 48%, #c7d8ef 100%);
        }

        [data-testid="stAppViewContainer"] .main { display: flex; justify-content: center; }
        [data-testid="stAppViewContainer"] .main > div { width: 100%; display: flex; justify-content: center; }

        .block-container {
            width: min(95vw, 1460px);
            margin: 92px auto 40px auto;
            padding: 0 0 22px 0;
            border-radius: 36px;
            border: 1px solid var(--shell-border);
            background: linear-gradient(132deg, rgba(255,255,255,.35), rgba(255,255,255,.16));
            box-shadow: 0 30px 52px rgba(80, 103, 152, .18);
            overflow: hidden;
            position: relative;
        }

        .block-container::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 170px;
            background: linear-gradient(180deg, rgba(177,198,235,0), rgba(177,198,235,.24));
            pointer-events: none;
        }

        .hero {
            padding: 30px 42px 28px 42px;
            border-bottom: 1px solid var(--shell-border);
            background:
                radial-gradient(circle at 52% -18%, rgba(255,255,255,.36), transparent 43%),
                linear-gradient(118deg, rgba(160,191,241,.22), rgba(233,212,241,.18), rgba(158,197,248,.22));
        }

        .hero-grid { display: grid; grid-template-columns: 330px 1fr; align-items: center; }
        .brand { display: inline-flex; align-items: center; gap: 12px; color: var(--txt); font-size: clamp(38px, 2.7vw, 56px); font-weight: 700; }
        .logo-svg { width: 62px; height: 62px; display: inline-flex; flex-shrink: 0; }
        .hero-copy { text-align: center; margin-right: 82px; }
        .hero-copy h1 { margin: 0; color: var(--txt); font-size: clamp(40px, 3.2vw, 62px); line-height: 1.1; font-weight: 700; }
        .hero-copy p { margin: 10px 0 0 0; color: var(--muted); font-size: clamp(17px, 1.25vw, 22px); font-weight: 500; }

        .content { padding: 30px 28px 28px 28px; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 32px;
            border: 1px solid var(--card-border);
            background: linear-gradient(130deg, rgba(255,255,255,.33), rgba(255,255,255,.16));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.40);
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-left),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-right) { min-height: 716px; padding: 11px; }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-left-inner) {
            border-radius: 30px; margin-top: 10px; padding: 14px; min-height: 530px;
            background: linear-gradient(130deg, rgba(255,255,255,.48), rgba(255,255,255,.26));
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-placeholder) {
            border-radius: 28px; margin-top: 8px; padding: 8px; min-height: 430px;
            background:
                radial-gradient(circle at 84% 84%, rgba(185, 216, 251, .30), transparent 40%),
                linear-gradient(130deg, rgba(255,255,255,.50), rgba(255,255,255,.26));
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.sale-box),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.rent-box),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.demand-box) {
            border-radius: 24px;
            margin-top: 8px;
            background: linear-gradient(130deg, rgba(255,255,255,.52), rgba(255,255,255,.30));
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.sale-box) {
            min-height: 188px;
            background:
                radial-gradient(circle at 88% 76%, rgba(190, 220, 252, .34), transparent 48%),
                linear-gradient(130deg, rgba(255,255,255,.56), rgba(255,255,255,.30));
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.rent-box) {
            min-height: 188px;
            background:
                radial-gradient(circle at 84% 82%, rgba(251, 207, 214, .24), transparent 48%),
                linear-gradient(130deg, rgba(255,255,255,.56), rgba(255,255,255,.30));
        }

        .panel-title { display: inline-flex; align-items: center; gap: 12px; color: var(--txt); font-size: clamp(32px, 2.1vw, 50px); line-height: 1.1; font-weight: 700; margin: 0; }

        .icon-bars { width: 30px; height: 26px; display: inline-flex; align-items: flex-end; gap: 3px; }
        .icon-bars span:nth-child(1) { width: 7px; height: 18px; border-radius: 2px; background: #55a8f8; }
        .icon-bars span:nth-child(2) { width: 7px; height: 24px; border-radius: 2px; background: #7cb8ff; }
        .icon-bars span:nth-child(3) { width: 7px; height: 20px; border-radius: 2px; background: #3f96f2; }

        .icon-bag {
            width: 32px; height: 32px; border-radius: 9px; background: linear-gradient(180deg, #f8c4c0, #ef9f9a);
            color: #fff; font-size: 20px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
            box-shadow: 0 6px 14px rgba(215, 142, 141, .30);
        }

        .dots { color: #b7c1de; font-size: 22px; font-weight: 700; letter-spacing: 4px; margin-right: 4px; }

        .await { text-align: center; padding: 12px 8px; }
        .await-row { display: flex; align-items: center; justify-content: center; gap: 18px; }
        .await-line { width: 108px; border-top: 5px dotted #ccd4e8; opacity: .72; }

        .await-bag {
            width: 112px; height: 124px; border-radius: 50px 50px 38px 38px;
            background: linear-gradient(180deg, #bfd6ff, #94b9f2);
            box-shadow: 0 14px 26px rgba(126, 159, 214, .32);
            position: relative; display: inline-flex; align-items: center; justify-content: center;
        }

        .await-bag::before {
            content: ""; position: absolute; top: -16px; width: 38px; height: 18px;
            border-radius: 10px; background: linear-gradient(180deg, #cde0ff, #a8c7fa);
        }

        .await-bag::after { content: "$"; color: #5f8fdc; font-size: 58px; font-weight: 800; }
        .await-title { margin-top: 20px; color: #465783; font-size: clamp(36px, 2.8vw, 56px); font-weight: 700; line-height: 1.1; }
        .await-desc { margin-top: 12px; color: #60719a; font-size: clamp(16px, 1.2vw, 20px); line-height: 1.45; }

        .result-card-sale, .result-card-rent { display: flex; align-items: center; justify-content: space-between; gap: 22px; padding: 4px 2px; }
        .sale-left { display: flex; align-items: center; gap: 16px; }

        .sale-icon {
            width: 104px; height: 114px; border-radius: 46px 46px 34px 34px;
            background: linear-gradient(180deg, #bfd6ff, #96bbf4);
            box-shadow: 0 12px 22px rgba(124, 158, 216, .30);
            position: relative; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
        }

        .sale-icon::before {
            content: ""; position: absolute; top: -14px; width: 34px; height: 16px; border-radius: 10px;
            background: linear-gradient(180deg, #cfe2ff, #acc9f9);
        }

        .sale-icon::after { content: "$"; color: #5f8fdc; font-size: 56px; font-weight: 800; }

        .result-caption { color: #3f4e79; font-size: clamp(20px, 1.7vw, 46px); font-weight: 500; line-height: 1.15; }
        .result-sale { margin-top: 6px; color: #2f4478; font-size: clamp(46px, 4vw, 84px); font-weight: 700; line-height: 1; }
        .result-rent { margin-top: 8px; color: #cf7781; font-size: clamp(44px, 3.8vw, 82px); font-weight: 700; line-height: 1; }

        .rent-house {
            width: 128px; height: 94px; border-radius: 12px; background: linear-gradient(180deg, #f8b7b5, #f2a6a4);
            position: relative; opacity: .9; flex-shrink: 0;
        }

        .rent-house::before {
            content: ""; position: absolute; left: -8px; right: -8px; top: -36px; height: 42px;
            background: linear-gradient(180deg, #f7b4b2, #f2a4a2); clip-path: polygon(50% 0, 100% 100%, 0 100%); border-radius: 10px;
        }

        .rent-house::after {
            content: "$"; position: absolute; left: 50%; top: 47%; transform: translate(-50%, -50%);
            color: #fff2f2; font-size: 56px; font-weight: 800;
        }

        .demand-wrap {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 8px 2px;
        }

        .demand-title {
            color: #3f4e79;
            font-size: 26px;
            font-weight: 600;
        }

        .demand-prob {
            color: #4f5f89;
            font-size: 20px;
            font-weight: 500;
            margin-top: 4px;
        }

        .demand-badge {
            padding: 10px 16px;
            border-radius: 999px;
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            box-shadow: 0 8px 16px rgba(0,0,0,.12);
            white-space: nowrap;
        }

        .demand-high { background: linear-gradient(180deg, #35c37d, #2aaf70); }
        .demand-low { background: linear-gradient(180deg, #ef6868, #d95454); }
        .demand-moderate { background: linear-gradient(180deg, #f3b24c, #e29b2d); }

        div[data-testid="stButton"] { margin-top: 14px; width: 100%; }
        div[data-testid="stButton"] > button {
            width: 100% !important; height: 86px; border: none !important; border-radius: 30px !important;
            font-size: 22px !important; font-weight: 700 !important; color: #f6fbff !important;
            background: linear-gradient(90deg, #f3a08e 0%, #9f8bde 45%, #72bcf8 100%) !important;
            box-shadow: 0 14px 24px rgba(90, 116, 186, .23) !important;
        }
        div[data-testid="stButton"] > button * { font-size: inherit !important; font-weight: inherit !important; }
        div[data-testid="stButton"] > button:hover { filter: brightness(1.03); }

        div[data-testid="stSlider"] > label, div[data-testid="stNumberInput"] > label {
            color: #3a4870 !important; font-size: 18px !important; font-weight: 600 !important;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(1),
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) { height: 9px !important; }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(1) { background: #d9dfef !important; }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) { background: linear-gradient(90deg, #5399f0, #8daeff) !important; }

        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background: #5c9df2 !important; border: 4px solid #e9effb !important;
            width: 30px !important; height: 30px !important; box-shadow: 0 6px 10px rgba(72, 108, 177, .24) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.age-area) {
            border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; margin: 0 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.age-area)
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
            background: linear-gradient(90deg, #5b9bf1, #f2a9b3) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.age-area)
        div[data-testid="stSlider"] [role="slider"] { background: #f0a7b3 !important; }

        div[data-testid="stNumberInput"] [data-baseweb="input"] {
            border-radius: 14px !important; border-color: rgba(172,186,217,.86) !important; background: rgba(255,255,255,.56) !important;
        }

        div[data-testid="stNumberInput"] input { color: #3a4972 !important; background: transparent !important; }

        div[data-testid="stNumberInput"] button {
            color: #5f7099 !important; border-color: rgba(172,186,217,.86) !important; background: rgba(255,255,255,.32) !important;
        }

        @media (max-width: 1100px) {
            .hero-grid { grid-template-columns: 1fr; gap: 12px; }
            .brand { justify-content: center; font-size: 38px; }
            .hero-copy { margin-right: 0; }
            .hero-copy h1 { font-size: 42px; }
            .hero-copy p { font-size: 18px; }
            .panel-title { font-size: 30px; }
            .await-title { font-size: 40px; }
            .await-desc { font-size: 18px; }
            .demand-title { font-size: 22px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_app() -> None:
    st.set_page_config(page_title="NestIQ Property Predictor", page_icon="🏠", layout="wide")
    _inject_styles()

    if "sale_price" not in st.session_state:
        st.session_state.sale_price = 645400.0
    if "rent_price" not in st.session_state:
        st.session_state.rent_price = 2800.0
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False
    if "demand_label" not in st.session_state:
        st.session_state.demand_label = "High Demand"
    if "demand_probability" not in st.session_state:
        st.session_state.demand_probability = "87%"

    st.markdown(
        """
        <div class="hero">
          <div class="hero-grid">
            <div class="brand">
              <span class="logo-svg" aria-hidden="true">
                <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M8 36L32 12L56 36H8Z" fill="#B5CDF4" fill-opacity="0.8"/>
                  <rect x="17" y="22" width="30" height="30" rx="8" transform="rotate(45 17 22)" fill="url(#g1)"/>
                  <circle cx="32" cy="32" r="8" fill="#EAF4FF"/>
                  <text x="32" y="35.5" text-anchor="middle" font-size="10" font-weight="800" fill="#5D97E8">$</text>
                  <defs>
                    <linearGradient id="g1" x1="32" y1="22" x2="32" y2="52" gradientUnits="userSpaceOnUse">
                      <stop stop-color="#71B5F8"/>
                      <stop offset="1" stop-color="#4B8EE8"/>
                    </linearGradient>
                  </defs>
                </svg>
              </span>
              NestIQ
            </div>
            <div class="hero-copy">
              <h1>Smart Property Intelligence</h1>
              <p>AI-Powered Property Valuation &amp; Market Demand Analysis</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="content">', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-left"></div>', unsafe_allow_html=True)
            h1, h2 = st.columns([10, 1])
            with h1:
                st.markdown(
                    '<div class="panel-title"><span class="icon-bars"><span></span><span></span><span></span></span>Property Features</div>',
                    unsafe_allow_html=True,
                )
            with h2:
                st.markdown('<div class="dots">•••</div>', unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="panel-left-inner"></div>', unsafe_allow_html=True)
                area = st.slider("Area (m²)", min_value=0, max_value=500, value=0)

                c1, c2 = st.columns(2)
                with c1:
                    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=0, step=1)
                with c2:
                    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=8, value=0, step=1)

                distance = st.slider("Distance from City Center (km)", min_value=0, max_value=30, value=0)

                with st.container(border=True):
                    st.markdown('<div class="age-area"></div>', unsafe_allow_html=True)
                    age = st.slider("Property Age (years)", min_value=0, max_value=80, value=0)

                all_features_zero = (
                    float(area) == 0.0
                    and float(bedrooms) == 0.0
                    and float(bathrooms) == 0.0
                    and float(distance) == 0.0
                    and float(age) == 0.0
                )
                has_required_room_inputs = float(bedrooms) > 0.0 and float(bathrooms) > 0.0
                can_predict = has_required_room_inputs and not all_features_zero

    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-right"></div>', unsafe_allow_html=True)
            h1, h2 = st.columns([10, 1])
            with h1:
                st.markdown(
                    '<div class="panel-title"><span class="icon-bag">$</span>Prediction Results</div>',
                    unsafe_allow_html=True,
                )
            with h2:
                st.markdown('<div class="dots">•••</div>', unsafe_allow_html=True)

            if not st.session_state.has_prediction:
                with st.container(border=True):
                    st.markdown('<div class="panel-placeholder"></div>', unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div class="await">
                          <div class="await-row">
                            <span class="await-line"></span>
                            <span class="await-bag"></span>
                            <span class="await-line"></span>
                          </div>
                          <div class="await-title">Awaiting Analysis</div>
                          <div class="await-desc">Your property insights will appear here<br>after running the AI prediction.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                with st.container(border=True):
                    st.markdown('<div class="sale-box"></div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="result-card-sale">
                          <div class="sale-left">
                            <div class="sale-icon"></div>
                            <div>
                              <div class="result-caption">Estimated Sale Price</div>
                              <div class="result-sale">${st.session_state.sale_price:,.0f}</div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.container(border=True):
                    st.markdown('<div class="rent-box"></div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="result-card-rent">
                          <div>
                            <div class="result-caption">Estimated Monthly Rent</div>
                            <div class="result-rent">${st.session_state.rent_price:,.0f}</div>
                          </div>
                          <div class="rent-house"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                badge_class = "demand-moderate"
                if st.session_state.demand_label == "High Demand":
                    badge_class = "demand-high"
                elif st.session_state.demand_label == "Low Demand":
                    badge_class = "demand-low"
                with st.container(border=True):
                    st.markdown('<div class="demand-box"></div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="demand-wrap">
                          <div>
                            <div class="demand-title">Demand Prediction</div>
                          </div>
                          <div class="demand-badge {badge_class}">{st.session_state.demand_label} · {st.session_state.demand_probability}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            predict_clicked = st.button(
                "◎  Predict Property Value",
                key="predict_button",
                use_container_width=True,
                disabled=not can_predict,
            )

            if not has_required_room_inputs:
                st.caption("Choose both Bedrooms and Bathrooms (greater than 0) to enable prediction.")
            elif all_features_zero:
                st.caption("Set at least one Property Feature above 0 to enable prediction.")

            if predict_clicked:
                features = torch.tensor(
                    [[float(area), float(bedrooms), float(bathrooms), float(distance), float(age)]],
                    dtype=torch.float32,
                )
                sale_price, rent_price, demand_label, demand_probability = predict_property(features)

                st.session_state.sale_price = sale_price
                st.session_state.rent_price = rent_price
                st.session_state.demand_label = demand_label
                st.session_state.demand_probability = demand_probability
                st.session_state.has_prediction = True
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_app()
