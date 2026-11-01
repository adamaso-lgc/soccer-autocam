# soccer-autocam

Câmara virtual (PTZ digital) que segue a bola num jogo de futebol e emite o resultado para um ficheiro ou para o YouTube (RTMP).

```
fonte (ficheiro / RTSP / webcam)
   -> detetor (YOLO: bola + jogadores)
   -> tracker da bola (associação, previsão, "coasting" quando se perde)
   -> câmara virtual (alvo = bola + centro dos jogadores, mola amortecida, zona morta, zoom)
   -> crop 16:9 sub-pixel -> ffmpeg (H.264) -> mp4 ou rtmp://youtube
```

Usa uma câmara fixa, grande angular e de alta resolução (4K+) e faz o "PTZ" por software: sem motores, sem latência mecânica. É a abordagem dos produtos comerciais.

## Estado atual

| Parte | Estado |
|---|---|
| Tracker, câmara virtual, suavização, saída mp4/RTMP, CLI | Implementado e testado (`pytest`, vídeo sintético) |
| `YoloDetector` (frame inteiro + tiles) | Implementado; lógica testada com modelo falso. **Ainda não validado com pesos reais nem em jogos reais** |
| Fine-tuning para bola de futebol | Por fazer (ver abaixo) |
| Duas câmaras (stitching) | Por fazer |
| PTZ físico (ONVIF) | Fora de âmbito por agora |

O detetor `demo` (`--detector demo`) só funciona no vídeo sintético: serve para validar o resto do pipeline sem GPU.

## Requisitos

- Python 3.10+
- ffmpeg (`sudo apt install ffmpeg` / `brew install ffmpeg`)
- Para tempo real: GPU NVIDIA com CUDA. Sem GPU consegues processar ficheiros offline (mais lento que o tempo real).

## Instalação

```bash
./scripts/setup.sh          # cria .venv, instala dependências, corre os testes
source .venv/bin/activate
```

Com GPU NVIDIA, instala primeiro o PyTorch com CUDA (ver comentário no `scripts/setup.sh`), senão o `pip` pode instalar a versão só CPU.

## 1. Demo sem câmara nem YOLO

```bash
./scripts/demo.sh
```

Gera um jogo sintético (2560x720, bola + 22 jogadores, com a bola a "desaparecer" 3 vezes) e produz:

- `outputs/demo_out.mp4`: o vídeo final 1080p que iria para o YouTube
- `outputs/demo_debug.mp4`: o vídeo original com overlay (verde = crop, amarelo = bola detetada, cruz vermelha = bola prevista, azul = jogadores)

## 2. Com um vídeo real + YOLO

Precisas de um vídeo de uma câmara fixa em posição alta que mostre o campo todo (não serve uma transmissão de TV, que muda de plano).

```bash
python -m autocam --source jogo.mp4 --output outputs/jogo_out.mp4 \
  --debug-video outputs/jogo_debug.mp4 --device cuda:0
```

Os pesos `yolov8m.pt` (COCO, classe `sports ball`) são descarregados automaticamente na primeira execução. Isto é só uma baseline: a bola tem poucos pixels em 4K e o COCO falha muito. Para melhorar:

1. **Tiles** para a bola: `--tile-size 1280` (corre o modelo em blocos sobrepostos; mais lento, apanha bolas pequenas).
2. **Baixar o limiar**: `--ball-conf 0.05` (o tracker filtra falsos positivos pela continuidade).
3. **Fine-tuning** num dataset de futebol com a bola anotada, depois `--model runs/detect/train/weights/best.pt`.
   Os repositórios SoccerNet que encontraste são sobretudo para ações/eventos; procura datasets com caixas da bola (por exemplo SoccerNet-Tracking, SoccerTrack) e confirma anotações e licença antes de treinar.
4. Quando a bola é perdida, a câmara segue o centro dos jogadores (fallback), por isso uma deteção medíocre ainda dá um resultado utilizável.

Aceleração: `--detect-every 2` (ou 3) corre o detetor de N em N frames; o tracker prevê entre deteções.

## 3. Câmara IP (RTSP)

```bash
python -m autocam --source "rtsp://user:pass@192.168.1.50:554/stream1" --output outputs/live.mp4
```

Em fontes live só se processa o frame mais recente (descarta frames se a inferência for lenta, em vez de acumular atraso). O transporte RTSP é forçado a TCP.

## 4. Emitir para o YouTube

1. YouTube Studio -> Criar -> Fazer uma emissão em direto -> copia a **chave de stream**.
2. Testa primeiro com um ficheiro, simulando tempo real:

```bash
python -m autocam --source jogo.mp4 --realtime --bitrate 6000k \
  --output "rtmp://a.rtmp.youtube.com/live2/CHAVE"
```

O vídeo sai a H.264, GOP de 2 s e áudio AAC silencioso (o YouTube espera uma pista de áudio). Precisas de upload estável, pelo menos 8-10 Mbps para 1080p a 6 Mbps. Não metas a chave de stream no git.

## Afinação

Parâmetros em `autocam/config.py` (e alguns na CLI):

| Parâmetro | Efeito |
|---|---|
| `smooth_time_pan` (`--smooth-pan`) | Mais alto = movimento mais suave/lento |
| `max_pan_speed` | Limite de velocidade da câmara |
| `dead_zone` | Evita micro-movimentos |
| `crop_width_ball` / `crop_width_wide` | Zoom com a bola localizada / perdida |
| `ball_weight` | Quanto a câmara favorece a bola vs. o grupo de jogadores |
| `lead_seconds` | Antecipa para onde a bola vai |
| `TrackerConfig.max_coast_s` | Quanto tempo prevê a bola sem a ver |

## Estrutura

```
autocam/
  config.py     parâmetros
  detectors.py  YoloDetector (+ tiles), DemoColorDetector
  framing.py    BallTracker, VirtualCamera, render_crop
  io.py         VideoSource (ficheiro/RTSP/webcam), FfmpegSink (mp4/RTMP)
  pipeline.py   loop principal
  synthetic.py  gerador do jogo sintético
  __main__.py   CLI
tests/          pytest (tracker, câmara, detetor com modelo falso, end-to-end)
scripts/        setup.sh, demo.sh
```

## Próximos passos sugeridos

1. Correr com um vídeo real de câmara fixa e olhar para o `--debug-video` para ver onde o detetor falha.
2. Fine-tuning do YOLO para a bola; exportar para TensorRT/ONNX para chegar a 25-30 fps.
3. Duas câmaras lado a lado: calibrar/colar (stitching) num panorama antes do detetor.
4. Estabilidade em direto: reconexão RTSP/RTMP, métricas de latência.
