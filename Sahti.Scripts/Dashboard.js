const pieCtx = document.querySelector('.pie-chart').getContext('2d');
const barCtx = document.querySelector('.bar-chart').getContext('2d');

new Chart(pieCtx, {
    type: 'pie',
    data: {
        labels: ['Cardiovascular', 'Oncological', 'Respiratory', 'Digestive', 'Neurological', 'Endocrine'],
        datasets: [{
            data: [28, 16, 13, 7, 6, 5], // Updated percentages from WHO and IHME
            backgroundColor: [
                'rgb(255,99,132)', 
                'rgb(54,162,235)', 
                'rgb(255,206,86)', 
                'rgb(75,192,192)', 
                'rgb(153,102,255)',
                'rgb(255,159,64)'
            ],
            borderWidth: 0,
            hoverBorderColor: 'rgb(100,100,100)',
            hoverBorderWidth: 3,
            hoverOffset: 50,
        }]
    },    
    options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
            padding: 10
        },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    color: 'white',
                    font: {
                        size: 20,
                        weight: 'bold'
                    },
                    boxWidth: 20,
                    padding: 10
                }
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                titleColor: 'rgb(255,255,255)',
                bodyColor: 'rgb(255,255,255)',
                bodyFont: {
                    size: 20
                },
                padding: 10,
                cornerRadius: 8
            },
            datalabels: {
                color: 'rgb(0,0,0)', 
                font: {
                    size: 20,
                    weight: 'bold'
                },
                formatter: (value, ctx) => {
                    let sum = ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                    let percentage = ((value / sum) * 100).toFixed(1) + '%';
                    return percentage;
                }
            }
        },
        animation: {
            animateScale: true, 
            animateRotate: true 
        }
    }
});


new Chart(barCtx, {
    type: 'bar',
    data: {
        labels: ['Gallstones', 'IBD', 'Celiac Disease', 'Lactose Intolerance', 'Gastroenteritis', 
                 'GERD', 'Diverticulosis and Diverticulitis', 'Peptic Ulcer Disease', 'Constipation', 'IBS'],
        datasets: [{
            label: 'Cases',
            data: [40, 50, 30, 60, 70, 120, 90, 80, 110, 60], // Randomized data for demonstration
            backgroundColor: [
                'rgb(255,99,132)', 
                'rgb(54,162,235)', 
                'rgb(255,206,86)', 
                'rgb(75,192,192)',               
                'rgb(153,102,255)', 
                'rgb(255,159,64)', 
                'rgb(54,235,87)', 
                'rgb(33,45,112)', 
                'rgb(255, 0, 0)', 
                'rgb(173,67,201)'
            ],                  
            hoverBackgroundColor: [
                'rgb(151, 59, 79)',
                'rgb(44, 112, 158)',
                'rgb(155, 124, 46)', 
                'rgb(53, 147, 147)',
                'rgb(86, 53, 152)',
                'rgb(150, 98, 47)', 
                'rgb(36, 136, 55)', 
                'rgb(0, 0, 0)', 
                'rgb(125, 16, 16)', 
                'rgb(116, 43, 136)'
            ],
            barThickness: 30
        }]
    },         
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                },
                ticks: {
                    color: 'rgb(255,255,255)',
                    font: {
                        size: 20
                    }
                }
            },
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: 'rgb(255,255,255)',
                    font: {
                        size: 14
                    }
                }
            }
        },
        plugins: {
            legend: {
                display: true,
                position: 'top'
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                titleColor: 'rgb(255,255,255)',
                bodyColor: 'rgb(255,255,255)'
            }
        }
    }
});
