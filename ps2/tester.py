def clean_data(x_vals, y_vals):
    dust = []
    efficiency = []
    for i in x_vals:
        try:
            float(i)
            dust.append(i)
        except ValueError:
            continue
    for j in y_vals:
        try:
            float(j)
            efficiency.append(j)
        except ValueError:
            continue

    return [dust,efficiency]
    raise NotImplementedError

print(clean_data(['0.47', '0.94', '1.41', '1.88', '2.35', '2.82', '3.228',"&*&34","*&2719"],['91.2', '70.5','0.47', '0.5%', '#4%.5', '0.94', '1.41', '1.88', '2.35', '$$']))
