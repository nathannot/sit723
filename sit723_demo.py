import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import datetime
import yfinance as yf
import SingleStockLogReward as logr
from dateutil.relativedelta import relativedelta

st.title('Sit723 Thesis Demo')

st.write('This app is a demo of the best algorithm based on Apple Stock')
st.write('You can test it from 2024 onwards on the stocks in the dropdown')

options = st.selectbox(
    "Select from the following Stocks",
    ("Apple", "Google", "Microsoft",'Amazon','Nvidia','Meta','Tesla'),
)

mapper = {'Apple':'aapl',
          'Google':'goog',
          'Microsoft':'msft',
          'Amazon':'amzn',
          'Nvidia':'nvda',
          'Meta':'meta',
          'Tesla':'tsla'
          }
inv = 10000

st.write('You can only select an end date that is at least 3 months from the start date.')
start = st.date_input('Select start date', value = datetime.date(2024,1,1),
                      min_value = datetime.date(2024,1,1))
min_end = start + relativedelta(months=3)
end = st.date_input('Select end date',value =datetime.date(2025,1,1),
                    min_value=min_end)

tk = yf.Ticker(mapper[options])

try:
    tk._fetch_ticker_tz(debug_mode=False, timeout=10)
except Exception as tz_err:
    st.warning(f"Warning: could not fetch timezone (proceeding anyway): {tz_err}")


v = tk.history(
    start=start,
    end=end,
    multi_level_index=False,
    progress=False
)
#v = yf.download(mapper[options], start, end, multi_level_index=False, session = session)
def calc_rsi(x, n=14):
    x = x.copy()
    diff = x.Close.diff()
    gains = diff.where(diff>0,0)
    losses = -diff.where(diff<0,0)
    m = len(gains)
    avg_gains = np.zeros(m)
    avg_losses = np.zeros(m)
    avg_gains[0] = np.sum(gains[:n])/n
    avg_losses[0] = np.sum(losses[:n])/n
    for i in range(1,m):
        avg_gains[i] = ((n-1)*avg_gains[i-1]+gains.iloc[i])/n
        avg_losses[i] = ((n-1)*avg_losses[i-1]+losses.iloc[i])/n

    RS = avg_gains/avg_losses
    RSI = 100 - 100/(1+RS)
    p = x.Close.values
    rsi_s = RSI/100
    rsi_p = rsi_s*(np.max(p)-np.min(p))+np.min(p)
    x['rsi'] = rsi_p
    return x

def create_bbands(x, n=30):
    x = x.copy()
    roll_mean = x.Close.rolling(n, min_periods=1).mean()
    roll_std = x.Close.rolling(n, min_periods=1).std()
    upper_bband = roll_mean+2*roll_std
    lower_bband = roll_mean - 2*roll_std
    x['UB'] = upper_bband
    x['LB'] = lower_bband
    return x.dropna()

def calc_obv(x):
    x = x.copy()
    n = len(x)
    p = x.Close.values
    vol = x.Volume.values
    obv = np.zeros(n)
    obv[0] = vol[0]
    direction = np.sign(np.diff(p))
    obv[1:] = obv[0]+np.cumsum(vol[1:]*direction)
    obv_s = (obv-np.min(obv))/(np.max(obv)-np.min(obv))
    obv_p = obv_s*(np.max(p)-np.min(p))+np.min(p)
    x['OBV'] = obv_p
    return x


def omrb(x, n=30):
    x = x.copy()
    x['MA'] = x.Close.rolling(n, min_periods=1).mean()
    x1 = calc_obv(x)
    x2 = calc_rsi(x1)
    x3 = create_bbands(x2)
    return x3.drop(['High','Open','Low','Volume'],axis=1)

valid = omrb(v)

env_logr_v = logr.SingleStockTraderLogReward(valid)
state_dim = env_logr_v.observation_space.shape[0]
action_dim = env_logr_v.action_space.n
num_atoms = 51
Vmin = -10
Vmax = 10
device = 'cpu'

def norm_state(x, f,l=30):
    y = x.reshape(f,l).T
    z=(y-y.min(axis=0))/(y.max(axis=0)-y.min(axis=0)+1e-5)
    return np.hstack(z.T)

class C51Net(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, Vmin=-10, Vmax=10, hid=64):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax - Vmin) / (num_atoms - 1)
        # Create the fixed support (atoms)
        self.register_buffer("support", torch.linspace(Vmin, Vmax, num_atoms))
        
        self.fc1 = nn.Linear(state_dim, hid)
        # Final layer outputs action_dim * num_atoms values
        self.fc2 = nn.Linear(hid, action_dim * num_atoms)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # Reshape to [batch, action_dim, num_atoms]
        x = x.view(-1, self.action_dim, self.num_atoms)
        # Apply softmax over atoms to get a probability distribution
        probabilities = F.softmax(x, dim=2)
        return probabilities
    

@st.cache_resource
def load_c51_checkpoint(path: str):
    # re-instantiate the exact same architecture
    model = C51Net(
        state_dim=state_dim,
        action_dim=action_dim,
        num_atoms=num_atoms,
        Vmin=Vmin,
        Vmax=Vmax,
        hid=64
    ).to(device)
    # load the weights from episode 200
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

ckpt_path = "c51_checkpoint_ep200.pth"
policy = load_c51_checkpoint(ckpt_path)

torch.manual_seed(4)

# Reset environment
state = env_logr_v.reset()[0]
state = norm_state(state, f=int(state.shape[0]/30))

done = False
episode_reward = 0
episode_history = []

# Single episode run using C51
while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # Get the distribution output: shape [1, action_dim, num_atoms]
    prob_dist = policy(state_tensor)
    # Compute expected Q-values: sum(prob * support) over atoms
    q_values = torch.sum(prob_dist * policy.support, dim=2)
    # Choose the action with the highest expected Q-value
    action = torch.argmax(q_values, dim=1).item()
    
    next_state, reward, done, _, info = env_logr_v.step(action)
    next_state = norm_state(next_state, f=int(state.shape[0]/30))
    episode_reward += reward
    episode_history.append(info)
    state = next_state

st.header('Holdings and Portfolio')
st.write('The graphs below show the share holding pattern the algorithm learns and the portfolio evolution of the chosen holding pattern against BnH')
fig, ax = plt.subplots(1, 2, figsize=(18, 5))
m_c51_logr_v = pd.DataFrame(episode_history)
ax[0].plot(m_c51_logr_v.Holdings)
ax[0].set_title("Holdings")
ax[0].set_ylabel('Number of Shares')

bnh = (inv * np.cumprod(1 + v[32:].Close.pct_change().dropna()))
ax[1].plot(bnh, label='BnH')
ax[1].plot(v.index[32:], m_c51_logr_v.Portfolio, label=f'{options} Portfolio')
plt.setp(ax[1].get_xticklabels(), rotation=45)
ax[1].legend()
ax[1].set_ylabel('Portfolio Value')
ax[1].set_title(f'Porfolio Growth on ${inv} investment')

st.pyplot(fig)
st.write(f'Final Portfolio value ${m_c51_logr_v.Portfolio.iloc[-1]:.2f}')
st.write(f'BnH final value ${bnh.iloc[-1]:.2f}')

st.write('This table summarises the number values of share holdings and portfolio value over time.')
st.write(m_c51_logr_v)
