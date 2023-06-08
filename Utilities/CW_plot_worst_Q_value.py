import matplotlib.pyplot as plt
import numpy as np
VI = np.array([-13.103336477142358, -14.210601816907863, -14.95491590134225, -15.965578595289982, -16.732281039116447, -17.643916507828433, -18.63424381746942, -19.61399155554886])
d0 = np.array([-35.677029807869374, -35.677029807869374, -39.5670517181297, -41.517131223313214, -43.76289762850427, -46.226191078381575, -45.20745829561165, -48.774166568596016])
d1 = np.array([-35.58697099706822, -36.09704388181479, -39.96444184243473, -40.66684136400322, -43.29185139108047, -44.83767332167736, -46.324839830591145, -48.34716497023022])

print(np.mean(-d0+VI))
print(np.mean(-d1+VI))


p_array = [0.025,0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
plt.plot(p_array, -VI, "o", color = "green", label = "Value iteration")
plt.plot(p_array, -d0, "o", color = "orange", label = "Online Non-DRQL")
plt.plot(p_array, -d1, "o", color = "blue", label = "Online DRQL")

plt.xlim(0.03,0.22)
plt.ylim(10,55)
plt.xlabel("Step-down values")
plt.ylabel("Cost")
plt.title("Plot of Value Iteration, Non-DRQL and DRQL")
plt.legend(loc="best")
plt.grid()
plt.show()
