from nodebox.graphics import *
from nodebox.graphics.physics import Vector, Boid, Flock, Obstacle

flock = Flock(50, x=-50, y=-50, width=700, height=400)
flock.sight(80)

def draw(canvas):
    canvas.clear()
    flock.update(separation=0.4, cohesion=0.6, alignment=0.1, teleport=True)
    for boid in flock:
        push()
        translate(boid.x, boid.y)
        scale(0.5 + boid.depth)
        rotate(boid.heading)
        arrow(0, 0, 15)
        pop()

canvas.size = 600, 300
canvas.run(draw)
