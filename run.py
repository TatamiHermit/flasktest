from flask import Flask, render_template
from flask import request
from flasktest.srcscript.polyline2gps import *

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def after_login():
    global src, des, vs_h, strategy, base_time
    if request.method == "GET":
        return render_template("gps.html", list = [], vs_h=60)
    elif request.method == "POST":
        src = request.form.get("src")
        des = request.form.get("des")
        vs = request.form.get("vs_h")
        base_time = request.form.get("base_time")
        strategy = request.form.get("strategy")
        if '' in [src, des, vs, base_time, strategy]:
            return '存在无效输入，请返回重试'
        else:
            vs_h = int(vs)
            write_log(f'{time.strftime("%Y_%m_%d_%H_%M")}_{src}_{des}_{vs_h}')
            get_polyline(src, des, strategy)
            get_gcj02_location()
            gcj02_interpolation(vs_h)
            list2 = get_linearr()
            get_wgs84_location()
            wgs84_interpolation(vs_h)
            datetime_to_msg(gps_date_can_id, base_time)
            angle_to_msg(gps_angle_can_id, elevation, vs_h)
            lon_to_msg(gps_lon_can_id)
            lat_to_msg(gps_lat_can_id)
            merge_message()
            msg_to_asc(src, des, vs_h)
            return render_template("gps.html", list = list2, vsh =vs_h )


if __name__ == '__main__':
    bus_interval_time = 0.1  # GPS信号帧间隔,单位秒
    gps_date_can_id = '351'  # GPS日期CAN ID
    gps_angle_can_id = '353'  # GPS航向角CAN ID
    gps_lat_can_id = '35C'  # GPS纬度CAN ID
    gps_lon_can_id = '35D'  # GPS经度CAN ID
    # vs_h = 400  # 车速, 单位千米/小时
    # src = "上海市" + "巨峰路2199号"  # 起点，可以是坐标点，如果是地址需要加上城市
    # des = "上海市" + "虹桥国际机场"  # 终点，可以是坐标点，如果是地址需要加上城市
    # strategy = '不走高速' #高速优先  躲避拥堵  不走高速  避免收费
    elevation = 400 #海拔m
    # base_time = "2020-01-17 8:05:50"  # 信号开始时间
    app.run()
    # ----------------以下为CAN总线报文数据处理---------------------------------------------#
    # -------------------------END---------------------------------------------------------#
    run_time()
