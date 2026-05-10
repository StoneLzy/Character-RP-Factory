# RAG Ask 测试问题清单

## 建议测试命令

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m crpf.cli rag-query --config config.yaml --top-k 5 "咲季和佑芽是什么关系？"
PYTHONPATH=src python3 -m crpf.cli rag-ask --config config.yaml --top-k 4 "咲季为什么害怕输给佑芽？"
```

## 15 个典型问题

1. 咲季和佑芽是什么关系？
2. 咲季为什么害怕输给佑芽？
3. 咲季和制作人的关系是什么？
4. 制作人为什么会选择培育咲季？
5. 咲季的核心性格是什么？
6. 咲季有哪些弱点或内在矛盾？
7. 咲季的成长弧线是怎样的？
8. 咲季的说话风格有什么特点？
9. 写咲季中文台词时应该避免哪些错误？
10. 咲季在 Re;IRIS 中承担什么角色？
11. Re;IRIS 对咲季的成长有什么影响？
12. 咲季和藤田琴音的关系是什么？
13. 咲季和月村手毬的关系是什么？
14. 初星学园的偶像培育机制有哪些重要设定？
15. 咲季如何面对失败、试镜和舞台压力？

## 批量手动测试

建议先串行跑，不要并发请求聊天模型：

```bash
while IFS= read -r q; do
  [ -z "$q" ] && continue
  echo
  echo "===== $q ====="
  PYTHONPATH=src python3 -m crpf.cli rag-ask --config config.yaml --top-k 4 "$q"
done <<'EOF'
咲季和佑芽是什么关系？
咲季为什么害怕输给佑芽？
咲季和制作人的关系是什么？
咲季的说话风格有什么特点？
Re;IRIS 对咲季的成长有什么影响？
EOF
```
