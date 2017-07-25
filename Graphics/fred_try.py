from graphics import *

def main():
    win = GraphWin("My Circle", 150, 100) # size of box
    c = Circle(Point(50,50), 30) # position of circle
    c.draw(win)
    win.getMouse() # Pause to view result
    win.close()    # Close window when done

main()
