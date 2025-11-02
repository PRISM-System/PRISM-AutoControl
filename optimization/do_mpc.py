import numpy as np

# AutoRegressive with eXogenous inputs : t-1 시점의 y와 x로 t 시점의 y를 예측하는 1차 선형 모델. do-mpc 라이브러리의 빠르고 간단한 제어를 위해 사용. 추후 수정 예정.
def fit_arx(df, x_col, y_col, window=300):
    # TIMESTAMP라는 column 명은 고정이라고 가정.. 아니면 쩔 수 없지
    cols = ['TIMESTAMP', y_col] + x_col
    dat = df[cols].tail(window).reset_index(drop=True).copy()
    y = dat[y_col].values.astype(float)
    X = []

    y1 = np.roll(y, 1); y1[0] = y1[1]
    X.append(y1)

    for mv in x_col:
        u = dat[mv].values.astype(float)
        u1 = np.roll(u, 1); u1[0] = u1[1]
        X.append(u1)

    X = np.column_stack(X + [np.ones_like(y)])
    X = X[2:]; y = y[2:]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(theta[0])
    b = [float(v) for v in theta[1:1+len(x_col)]]
    c = float(theta[-1])
    last = df.tail(1).iloc[0]
    return a, np.array(b), c, last

# MPC 예측
def MPC_predictions(mpc, x_now_scalar, extracted_X):
    x_now = np.array([[float(x_now_scalar)]], dtype=float)
    u_star = mpc.make_step(x_now) 
    x_pred = np.array(mpc.data.prediction(('_x', 'x'))).squeeze()
    u_cols = []
    for mv in extracted_X:
        u_mv = np.array(mpc.data.prediction(('_u', mv))).squeeze()
        u_cols.append(u_mv)
    u_pred = np.stack(u_cols, axis=1) if len(u_cols) > 0 else np.zeros((x_pred.shape[0]-1, 0))

    return x_pred[:-1], u_pred