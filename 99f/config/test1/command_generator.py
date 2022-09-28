def generate_command(prompt_index, seed, strength, image):
    command = f"--latent_0 {prompt_index} -I {image} -f {strength}"
    if seed is not None:
        command += f" -S {seed}"
    return command

def generate_batch(prompt_index, seed, strength, images):
    return [generate_command(prompt_index, seed, strength, image) for image in images]

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
    commands = commands + generate_batch(0, 2639741, 0.05, images_small) # clip 1
    commands = commands + generate_batch(0, 2639741, 0.1, images_small) # clip 2
    commands = commands + generate_batch(0, 2639741, 0.2, images_small) # clip 3
    commands = commands + generate_batch(0, 2639741, 0.3, images_small) # clip 4
    commands = commands + generate_batch(0, 2639741, 0.4, images_small) # clip 5
    commands = commands + generate_batch(0, 2639741, 0.5, images_small) # clip 6
    commands = commands + generate_batch(1, 2639741, 0.2, images_small) # clip 7
    commands = commands + generate_batch(1, 2639741, 0.5, images_small) # clip 8
    commands = commands + generate_batch(2, 2639741, 0.2, images_small) # clip 9
    commands = commands + generate_batch(3, 2639741, 0.2, images_small) # clip 10
    commands = commands + generate_interpolated_batch(0, 1, 0.0, 0.5, 9087143, images_large) # clip 11
    commands = commands + generate_interpolated_batch(2, 3, 0.0, 0.5, 9875692, images_large) # clip 12
    commands = commands + generate_batch(0, None, 0.1, images_small) # clip 13
    commands = commands + generate_batch(0, None, 0.2, images_small) # clip 14
    commands = commands + generate_batch(0, None, 0.3, images_small) # clip 15
    commands = commands + generate_batch(0, None, 0.4, images_small) # clip 16
    commands = commands + generate_batch(0, None, 0.5, images_small) # clip 17

    with open("commands.txt", "w") as f:
        f.write("\n".join(commands))
