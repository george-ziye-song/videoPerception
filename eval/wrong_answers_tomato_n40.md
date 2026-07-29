# TOMATO n=40 答错题目汇总（跨4个预算配置：2048/4096/8192/32768）

模型：Qwen3.5-9B，官方采样参数+标准化输出格式+reasoning_tags none

总共29条题目在至少一个配置下答错（满分40条）

| doc_id | 答错次数(共4个配置) | 哪些配置答错 | 题目 | 正确答案 | 视频路径 |
|---|---|---|---|---|---|
| 0 | 4/4 | 2048,4096,8192,32768 | In which direction(s) did the person's hand move? | C. Right. | videos/human/0231-04.mp4 |
| 2 | 4/4 | 2048,4096,8192,32768 | In which direction(s) did the person's hand move? | C. Left. | videos/human/0231-06.mp4 |
| 5 | 4/4 | 2048,4096,8192,32768 | In which direction(s) did the person's hand move? | C. Right. | videos/human/0221-04.mp4 |
| 7 | 4/4 | 2048,4096,8192,32768 | In which direction(s) did the person's hand move? | A. Right. | videos/human/0223-04.mp4 |
| 8 | 4/4 | 2048,4096,8192,32768 | In which direction(s) did the person's hand move? | E. Left. | videos/human/0223-05.mp4 |
| 9 | 4/4 | 2048,4096,8192,32768 | In which direction(s) did the person's hand move? | A. Right. | videos/human/0224-04.mp4 |
| 10 | 4/4 | 2048,4096,8192,32768 | What was the direction of the person's hand movement? | E. Left. | videos/human/0224-05.mp4 |
| 15 | 4/4 | 2048,4096,8192,32768 | What was the direction of the person's hand movement? | C. Right. | videos/human/0227-05.mp4 |
| 22 | 4/4 | 2048,4096,8192,32768 | What is the direction of the person's hand movement? | B. Leftwards then rightwards. | videos/human/0524-00.mp4 |
| 26 | 4/4 | 2048,4096,8192,32768 | What is the direction of the person's hand movement? | D. Leftwards then leftwards. | videos/human/0531-00.mp4 |
| 27 | 4/4 | 2048,4096,8192,32768 | What is the direction of the person's hand movement? | B. Leftwards then leftwards. | videos/human/0531-01.mp4 |
| 1 | 3/4 | 4096,8192,32768 | In which direction(s) did the person's hand move? | E. First to the left then to the right. | videos/human/0231-05.mp4 |
| 3 | 3/4 | 4096,8192,32768 | In which direction(s) did the person's hand move? | D. Up. | videos/human/0234-02.mp4 |
| 6 | 3/4 | 4096,8192,32768 | In which direction(s) did the person's hand move? | E. Left. | videos/human/0221-05.mp4 |
| 14 | 3/4 | 2048,4096,32768 | What was the direction of the person's hand movement? | C. Left. | videos/human/0227-04.mp4 |
| 19 | 3/4 | 2048,8192,32768 | What was the direction of the person's hand movement? | D. Down. | videos/human/0229-05.mp4 |
| 29 | 3/4 | 2048,8192,32768 | What is the direction of the person's hand movement? | E. Leftwards then upwards. | videos/human/0534-01.mp4 |
| 32 | 3/4 | 4096,8192,32768 | In which way did the person's hand move? | E. Upwards then Upwards. | videos/human/0537-00.mp4 |
| 16 | 2/4 | 2048,8192 | What was the direction of the person's hand movement? | D. Down. | videos/human/0228-04.mp4 |
| 23 | 2/4 | 2048,32768 | What is the direction of the person's hand movement? | C. Leftwards then rightwards. | videos/human/0524-01.mp4 |
| 24 | 2/4 | 4096,8192 | What is the direction of the person's hand movement? | A. Leftwards then downwards. | videos/human/0529-00.mp4 |
| 33 | 2/4 | 2048,8192 | In which way did the person's hand move? | C. Upwards then Upwards. | videos/human/0537-01.mp4 |
| 11 | 1/4 | 32768 | What was the direction of the person's hand movement? | C. Down. | videos/human/0225-04.mp4 |
| 13 | 1/4 | 2048 | What was the direction of the person's hand movement? | B. Left. | videos/human/0226-03.mp4 |
| 25 | 1/4 | 4096 | What is the direction of the person's hand movement? | A. Leftwards then downwards. | videos/human/0529-01.mp4 |
| 28 | 1/4 | 2048 | What is the direction of the person's hand movement? | A. Leftwards then upwards. | videos/human/0534-00.mp4 |
| 35 | 1/4 | 2048 | In which way did the person's hand move? | D. Upwards then rightwards. | videos/human/0539-01.mp4 |
| 38 | 1/4 | 32768 | In which way did the person's hand move? | F. Rightwards then upwards. | videos/human/0542-00.mp4 |
| 39 | 1/4 | 32768 | In which way did the person's hand move? | C. Rightwards then upwards. | videos/human/0542-01.mp4 |