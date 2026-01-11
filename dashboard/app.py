import streamlit as st
import os
st.set_page_config(page_title="Wordle模型实验结果", layout="wide")

# 页面标题
st.title("Wordle模型实验结果Dashboard")

# 定义图片路径字典
image_paths = {
    "data_source": {
        "games_char_embedding_distribution": "visualizations/data_source/games_char_embedding_distribution.png",
        "games_difficulty_distribution": "visualizations/data_source/games_difficulty_distribution.png",
        "games_steps_distribution": "visualizations/data_source/games_steps_distribution.png",
        "random_difficulty_distribution": "visualizations/data_source/random_difficulty_distribution.png",
        "random_steps_distribution": "visualizations/data_source/random_steps_distribution.png"
    },
    "LSTM_four_stage": {
        "learning_curves": "visualizations/LSTM_four_stage/learning_curves .png",
        "mae_across_stages": "visualizations/LSTM_four_stage/mae_across_stages.png",
        "mae_improvement_after_reinforcement": "visualizations/LSTM_four_stage/mae_improvement_after_reinforcement.png",
        "mse_across_stages": "visualizations/LSTM_four_stage/mse_across_stages.png",
        "mse_improvement_after_reinforcement": "visualizations/LSTM_four_stage/mse_improvement_after_reinforcement.png"
    },
    "LSTM_embedding_single_stage": {
        "learning_curves": "visualizations/LSTM_embedding_single_stage/learning_curves.png",
        "mae_across_stages": "visualizations/LSTM_embedding_single_stage/mae_across_stages.png",
        "mae_trend": "visualizations/LSTM_embedding_single_stage/mae_trend.png",
        "mse_across_stages": "visualizations/LSTM_embedding_single_stage/mse_across_stages.png"
    },
    "Transformer_four_stage": {
        "learning_curves": "visualizations/Transformer_four_stage/learning_curves.png",
        "mae_across_stages": "visualizations/Transformer_four_stage/mae_across_stages.png",
        "mae_improvement_after_reinforcement": "visualizations/Transformer_four_stage/mae_improvement_after_reinforcement.png",
        "mse_across_stages": "visualizations/Transformer_four_stage/mse_across_stages.png",
        "mse_improvement_after_reinforcement": "visualizations/Transformer_four_stage/mse_improvement_after_reinforcement .png"
    },
    "Transformer_single_stage": {
        "mae_across_stages": "visualizations/Transformer_single_stage/mae_across_stages.png",
        "mse_across_stages": "visualizations/Transformer_single_stage/mse_across_stages.png",
        "training_loss_curve": "visualizations/Transformer_single_stage/training_loss_curve.png"
    }
}

# 数据处理
def show_data_processing():
    st.header("1. 数据处理")
    st.write("这部分展示了Wordle游戏数据的处理结果，包括原始数据的分布情况。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("游戏数据分布")
        st.image(image_paths["data_source"]["games_steps_distribution"], caption="游戏步骤分布")
        st.image(image_paths["data_source"]["games_difficulty_distribution"], caption="游戏难度分布")
    
    with col2:
        st.subheader("随机数据分布")
        st.image(image_paths["data_source"]["random_steps_distribution"], caption="随机步骤分布")
        st.image(image_paths["data_source"]["random_difficulty_distribution"], caption="随机难度分布")
    
    st.subheader("字符嵌入分布")
    st.image(image_paths["data_source"]["games_char_embedding_distribution"], caption="游戏字符嵌入分布")

# LSTM四阶段部分
def show_lstm_four_stage():
    st.header("2. LSTM四阶段")
    st.write("这部分展示了LSTM四阶段模型的实验结果，包括学习曲线、MAE和MSE指标在各阶段的表现。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("学习曲线")
        st.image(image_paths["LSTM_four_stage"]["learning_curves"], caption="学习曲线")
        st.image(image_paths["LSTM_four_stage"]["mae_across_stages"], caption="各阶段MAE表现")
    
    with col2:
        st.subheader("阶段间指标对比")
        st.image(image_paths["LSTM_four_stage"]["mse_across_stages"], caption="各阶段MSE表现")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("强化学习后MAE改进")
        st.image(image_paths["LSTM_four_stage"]["mae_improvement_after_reinforcement"], caption="强化学习后MAE改进")
    
    with col4:
        st.subheader("强化学习后MSE改进")
        st.image(image_paths["LSTM_four_stage"]["mse_improvement_after_reinforcement"], caption="强化学习后MSE改进")

# LSTM单阶段部分
def show_lstm_single_stage():
    st.header("3. LSTM单阶段")
    st.write("这部分展示了LSTM单阶段模型的实验结果，包括学习曲线、MAE和MSE指标在各阶段的表现。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("学习曲线")
        st.image(image_paths["LSTM_embedding_single_stage"]["learning_curves"], caption="学习曲线")
        st.image(image_paths["LSTM_embedding_single_stage"]["mae_across_stages"], caption="各阶段MAE表现")
    
    with col2:
        st.subheader("阶段间指标对比")
        st.image(image_paths["LSTM_embedding_single_stage"]["mse_across_stages"], caption="各阶段MSE表现")
        st.image(image_paths["LSTM_embedding_single_stage"]["mae_trend"], caption="MAE趋势")

# Transformer四阶段部分
def show_transformer_four_stage():
    st.header("4. Transformer四阶段")
    st.write("这部分展示了Transformer四阶段模型的实验结果，包括学习曲线、MAE和MSE指标在各阶段的表现。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("学习曲线")
        st.image(image_paths["Transformer_four_stage"]["learning_curves"], caption="学习曲线")
        st.image(image_paths["Transformer_four_stage"]["mae_across_stages"], caption="各阶段MAE表现")
    
    with col2:
        st.subheader("阶段间指标对比")
        st.image(image_paths["Transformer_four_stage"]["mse_across_stages"], caption="各阶段MSE表现")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("强化学习后MAE改进")
        st.image(image_paths["Transformer_four_stage"]["mae_improvement_after_reinforcement"], caption="强化学习后MAE改进")
    
    with col4:
        st.subheader("强化学习后MSE改进")
        st.image(image_paths["Transformer_four_stage"]["mse_improvement_after_reinforcement"], caption="强化学习后MSE改进")

# Transformer单阶段部分
def show_transformer_single_stage():
    st.header("5. Transformer单阶段")
    st.write("这部分展示了Transformer单阶段模型的实验结果，包括学习曲线、MAE和MSE指标在各阶段的表现。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("训练损失曲线")
        st.image(image_paths["Transformer_single_stage"]["training_loss_curve"], caption="训练损失曲线")
    
    with col2:
        st.subheader("各阶段MAE表现")
        st.image(image_paths["Transformer_single_stage"]["mae_across_stages"], caption="各阶段MAE表现")
    
    st.subheader("各阶段MSE表现")
    st.image(image_paths["Transformer_single_stage"]["mse_across_stages"], caption="各阶段MSE表现")

if __name__ == "__main__":
    st.sidebar.title("导航菜单")
    page = st.sidebar.radio(
        "选择要查看的部分:",
        ["数据处理", "LSTM四阶段", "LSTM单阶段", "Transformer四阶段", "Transformer单阶段"]
    )
    if page == "数据处理":
        show_data_processing()
    elif page == "LSTM四阶段":
        show_lstm_four_stage()
    elif page == "LSTM单阶段":
        show_lstm_single_stage()
    elif page == "Transformer四阶段":
        show_transformer_four_stage()
    elif page == "Transformer单阶段":
        show_transformer_single_stage()
