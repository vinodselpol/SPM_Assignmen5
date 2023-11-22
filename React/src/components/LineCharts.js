import React from "react";

import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

// Refer the high charts "https://www.highcharts.com/demo/bar-basic" for more information

const LineCharts = (props) => {
  const config = {
    chart: {
      type: "line",
    },
    title: {
      text: props.title,
    },
    xAxis: {
      type: "category",
      labels: {
        rotation: -45,
        style: {
          fontSize: "13px",
          fontFamily: "Verdana, sans-serif",
        },
      },
    },
    yAxis: {
      min: 0,
      title: {
        text: props.y_axis,
      },
    },
    legend: {
      enabled: false,
    },
    tooltip: {
      pointFormat: props.y_axis + ": <b>{point.y} </b>",
    },
    series: [
      {
        name: props.title,
        data: props.data,
        dataLabels: {
          enabled: true,
          rotation: 0,
          color: "#FFFFFF",
          align: "center",
          format: "{point.y}", // one decimal
          y: 10, // 10 pixels down from the top
          style: {
            fontSize: "13px",
            fontFamily: "Verdana, sans-serif",
          },
        },
      },
    ],
  };
  return (
    <div>
      <div>
        <HighchartsReact
          highcharts={Highcharts}
          options={config}
        ></HighchartsReact>
      </div>
    </div>
  );
};

export default LineCharts;