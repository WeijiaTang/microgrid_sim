# Data Layout

当前仓库的 `data/` 目录按 **network-first 主线** 与 **legacy 聚合级资产** 分层：

```text
data/
  raw/
    weather/
      oikolab/
        los_angeles/
          los_angeles_hourly_2023.csv
          los_angeles_hourly_2024.csv
          los_angeles_hourly_2023_2024.csv
  processed/
    reference_15min/
      los_angeles_2023_2024/
        weather_15min.csv
        pv_reference_15min.csv
        load_reference_15min.csv
        price_reference_15min.csv
        metadata.json
    network_15min/
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

- `raw/` 保存未缩放的原始气象数据，目前采用洛杉矶 2023-2024 Oikolab 小时天气作为项目原始天气层
- `processed/reference_15min/` 保存两年 15 分钟参考天气、PV、负荷和价格序列
- `processed/network_15min/` 是当前网络环境、MILP、GA 与训练脚本共享的 canonical case 数据层
- `legacy/aggregated/` 与 `legacy/yearly/` 只服务旧 `MG-RES / MG-CIGRE` 数据读取
- `processed/network_15min/<case>/` 下的 `load.csv / pv.csv / price.csv` 是 `NetworkMicrogridEnv` 与 `load_network_profiles()` 的默认输入
- 若 `processed/network_15min/<case>/` 下没有 profile，当前网络环境才会回退到代码内的合成时序生成器
- 当前仓库已内置 `cigre_eu_lv/` 与 `ieee33/` 的 canonical 两年 15 分钟时序数据，时间范围为 2023-01-01 00:00:00 到 2024-12-31 23:45:00，每个序列长度为 `70176`
- 这两套网络数据由洛杉矶 2023-2024 原始小时气象、参考负荷层和 `legacy/yearly/tariff/price_profile.csv` 的 hourly TOU tariff 重建后，再按各自网络 case 的 `load_max_power` 与 `pv_max_power` 进行缩放
- `data/network/` 已退役，不再作为 canonical 数据层保留
