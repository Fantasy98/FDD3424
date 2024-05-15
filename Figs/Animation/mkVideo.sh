#!/
ffmpeg -framerate 10 -i Manifold_%05d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p manifold.mp4
ffmpeg -framerate 5 -i VAE_VS_POD_%05d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p vae_vs_pod.mp4
