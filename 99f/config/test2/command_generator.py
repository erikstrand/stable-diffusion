def generate_command(outdir, prompt_index, seed, strength, image):
    command = f"--latent_0 {prompt_index} -I {image} -f {strength} -W640 -H320 --outdir {outdir}"
    if seed is not None:
        command += f" -S {seed}"
    return command

def generate_batch(outdir, prompt_index, seed, strength, images):
    return [generate_command(outdir, prompt_index, seed, strength, image) for image in images]

def generate_interpolated_command(p0, p1, s0, s1, u, seed, image):
    f = (1.0 - u) * s0 + u * s1
    return f"--latent_0 {p0} --latent_1 {p1} -u {u:.5f} -I {image} -f {f} -S {seed}"

def generate_interpolated_batch(p0, p1, s0, s1, seed, images):
    commands = []
    for idx, img in enumerate(images):
        u = float(idx) / (len(images) - 1)
        commands.append(generate_interpolated_command(p0, p1, s0, s1, u, seed, img))
    return commands

if __name__ == "__main__":
    images_small = [f"./99f/mermaid_1/frames_1/IM{i:05d}.jpg" for i in range(0, 251)]
    images_large = [f"./99f/mermaid_1/frames_1/IM{i:05d}.jpg" for i in range(0, 751)]

    commands = []
    commands = commands + generate_batch(0, 2639741, 0.01, images_small) # clip 0

    with open("commands.txt", "w") as f:
        f.write("\n".join(commands))
