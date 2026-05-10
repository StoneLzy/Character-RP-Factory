# Raw CSV

当前原始 CSV 已经放在项目根目录的 `CSV/` 下。为了避免复制大量数据，默认配置 `paths.raw_csv_dir` 指向 `CSV`。

如果以后要把原始数据纳入标准目录，可以把 `CSV/` 移到这里，或在 `config.yaml` 中改成：

```yaml
paths:
  raw_csv_dir: data/raw_csv
```
