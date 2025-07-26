import os
import argparse

def rerwrite_svg_from_web(init_svg_path, new_svg_path):
    with open(init_svg_path, 'r') as f:
        lines = f.readlines()

    with open(new_svg_path, 'w') as f:
        f.write('<svg height="256" version="1.1" width="256" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('  <defs/>\n')
        f.write('  <g>\n')
        for line in lines:
            if line[:10] == '<path    s':
                stroke_mark = 0
                stroke_traj_place = line.find('d=')
                if stroke_traj_place < 0:
                    print('error' + init_svg_path)
                else:
                    stroke_traject = line[stroke_traj_place:-2]

                for l in line.split():
                    if l[:12] == 'stroke-width':
                        stroke_width = l
                        stroke_mark += 1
                for l in line.split('"'):
                    if l[:4] == 'rgba':
                        stroke_color = 'stroke="rgb(' + ', '.join(l[5:-1].split(', ')[:-1]) + ')"'
                if stroke_mark != 1:
                    print('error' + init_svg_path)

                f.write('    <path ')
                f.write(stroke_traject)
                f.write(' ')
                f.write('fill="none" ')
                f.write(stroke_color)
                f.write(' stroke-linecap="round" stroke-linejoin="round" stroke-opacity="1.0" ')
                f.write(stroke_width)
                f.write('/>\n')
        f.write('  </g>\n')
        f.write('</svg>\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg_path", type=str, required=True)
    args = parser.parse_args()
    origin_file = args.svg_path
    save_file = args.svg_path
    rerwrite_svg_from_web(origin_file, save_file)

main()
