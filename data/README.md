# Data Layout

当前仓库的 `data/` 目录按 **network-first 主线** 与 **legacy 聚合级资产** 分层：

```text
data/
  network/
    cigre_eu_lv/
      load.csv
      pv.csv
      price.csv
    ieee33/
      load.csv
      pv.csv
      price.csv
  legacy/
    aggregated/
      mg_res/
      mg_cigre/
    yearly/
      load/
      pv/
      weather/
      tariff/
```

规则：

- `network/` 是当前主路径，对应 `NetworkMicrogridEnv` 与 `load_network_profiles()`
- `legacy/aggregated/` 与 `legacy/yearly/` 只服务旧 `MG-RES / MG-CIGRE` 数据读取
- `network/<case>/` 下的 `load.csv / pv.csv / price.csv` 若存在，会优先覆盖合成 profile
- 若 `network/<case>/` 下没有 profile，当前网络环境会回退到代码内的合成时序生成器
