
import apa102
import time
from gpiozero import LED
try:
    import queue as Queue
except ImportError:
    import Queue as Queue



COLORS_RGB = dict(
    blue=(0, 0, 17),
    green=(0, 17, 0),
    orange=(17, 9, 0),
    pink=(17, 3, 10),
    purple=(9, 0, 9),
    red=(17, 0, 0),
    white=(17, 17, 17),
    yellow=(17, 17, 3),
    newcolor=(0,8,16),
    newcolor1=(0,12,12),
    off=(0, 0, 0),
)


class Pixels:
    PIXELS_N = 12

    def __init__(self):
        self.dev = apa102.APA102(num_led=self.PIXELS_N)
        self.power = LED(5)
        self.power.on()
        self.status = "off"
        
    def listen(self):
        color1 = COLORS_RGB['purple']
        color2 = COLORS_RGB['newcolor']
        for i in range(12):
                self.dev.set_pixel(i, color1[0], color1[1], color1[2])
        self.show()
        
    def processing(self):
        color1 = COLORS_RGB['blue']
        color2 = COLORS_RGB['newcolor1']
        for i in range(12):
            if i % 2 == 0:
                self.dev.set_pixel(i, color1[0], color1[1], color1[2])
            else:
                self.dev.set_pixel(i, color2[0], color2[1], color2[2])
        self.show()

        while self.status == "process":
            temp = color1
            color1 = color2
            color2 = temp
            self.show()
            time.sleep(0.2)
            for i in range(12):
                if i % 2 == 0:
                    self.dev.set_pixel(i, color1[0], color1[1], color1[2])
                else:
                    self.dev.set_pixel(i, color2[0], color2[1], color2[2])
                    
            if self.status != "process":
                break
        
        # Clear the LED pattern for the new status
        self.clear_strips()
        if self.status == "off":
            self.off()
        elif self.status == "wake up":
            self.wake_up()
        elif self.status == "speak":
            self.speak()
        elif self.status == "listen":
            self.listen()
 
 
            
    def speak(self):
        temp = 0
        color1 = COLORS_RGB['purple']
        color2 = COLORS_RGB['newcolor1']
        for i in range(12):
            if i % 2 == 0:
                self.dev.set_pixel(i, color1[0], color1[1], color1[2])
            else:
                self.dev.set_pixel(i, color2[0], color2[1], color2[2])
        self.show()
        time.sleep(0.2)
        
        while self.status == "speak":
            
            for i in range(12):
                if i % 2 != 0:
                    self.dev.set_pixel(i, 0, 0, 0)
                self.show()
            time.sleep(0.5)
            for i in range(12):
                if i % 2 == 0:
                    self.dev.set_pixel(i, color1[0], color1[1], color1[2])
                else:
                    self.dev.set_pixel(i, color2[0], color2[1], color2[2])
            self.show()
            time.sleep(0.5)
            
            for i in range(12):
                if i % 2 == 0:
                    self.dev.set_pixel(i, 0, 0, 0)

            self.show()
            time.sleep(0.5)
            
            for i in range(12):
                if i % 2 == 0:
                    self.dev.set_pixel(i, color1[0], color1[1], color1[2])
                else:
                    self.dev.set_pixel(i, color2[0], color2[1], color2[2])
            self.show()
            time.sleep(0.5)
            
            if self.status != "speak":
                break
                
        self.clear_strips()
        if self.status == "off":
            self.off()
        elif self.status == "wake up":
            self.wake_up()
        elif self.status == "process":
            self.processing()
        elif self.status == "listen":
            self.listen()
                
            
    def wake_up(self):
        color = COLORS_RGB['newcolor']
        for i in range(1,12):
            self.dev.set_pixel(i, color[0], color[1], color[2])
            self.show()
            time.sleep(0.35)   
        self.dev.set_pixel(0, 0, 24, 0)
        self.show()
            
    def off(self):
        for i in range (12):
            self.dev.set_pixel(i,0,0,0)
        self.show()
        
    def update_status(self, status):
        self.status = status
        print (status)
        
    def clear_strips(self):
        self.dev.clear_strip()
        
        
    def show(self):
        for i in range(self.PIXELS_N):
         

          self.dev.show()
          


