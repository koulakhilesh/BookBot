# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:23:53 2021

@author: akhilesh.koul
"""


from flask import Flask,render_template
app = Flask(__name__)

posts=[{
        'author': 'Akhilesh',
        'title': 'New Blog',
        'content': 'BookBot',    
        'date_posted': 'October 28, 2021',
        },
       {
        'author': 'Koul',
        'title': 'New Blog 2',
        'content': 'BookBot 2',    
        'date_posted': 'November 15, 2021',
        },
       ]

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return  render_template('about.html',title='About')


if __name__ == '__main__':
    app.run(debug=True)
