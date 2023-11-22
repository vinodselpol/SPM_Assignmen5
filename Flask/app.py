'''
Goal of Flask Microservice:
1. Flask will take the repository_name such as angular, angular-cli, material-design, D3 from the body of the api sent from React app and 
   will utilize the GitHub API to fetch the created and closed issues. Additionally, it will also fetch the author_name and other 
   information for the created and closed issues.
2. It will use group_by to group the data (created and closed issues) by month and will return the grouped data to client (i.e. React app).
3. It will then use the data obtained from the GitHub API (i.e Repository information from GitHub) and pass it as a input request in the 
   POST body to LSTM microservice to predict and forecast the data.
4. The response obtained from LSTM microservice is also return back to client (i.e. React app).

Use Python/GitHub API to retrieve Issues/Repos information of the past 1 year for the following repositories:
- https: // github.com/angular/angular
- https: // github.com/angular/material
- https: // github.com/angular/angular-cli
- https: // github.com/d3/d3
'''
# Import all the required packages 
import os
import datetime
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import pandas as pd
import requests

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

# Add response headers to accept all types of  requests
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Modify response headers when returning to the origin
def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

def convert_to_day(param):
    day, month, year = (int(x) for x in param.split('-'))    
    ans = datetime.date(day, month, year)
    return ans.strftime("%A")

#create a repositories to iterate the stars and count
repositories = ["angular/angular","openai/openai-cookbook", "openai/openai-python", "openai/openai-quickstart-python","milvus-io/pymilvus", "SeleniumHQ/selenium", "golang/go", "google/go-github","angular/material","angular/angular-cli","SebastianM/angular-googlemaps","d3/d3","facebook/react","tensorflow/tensorflow","keras-team/keras","pallets/flask"]
names =["Angular","Openai-cookbook","Openai-python","Openai-quickstart-python","Pymilvus","Selenium","Go","Go-github","Material","Angular-cli","Angular-googlemaps","D3","React","Tensorflow","Keras","Flask"]
'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/api/github', methods=['POST'])
def github():
    body = request.get_json()
    # Extract the choosen repositories from the request
    repo_name = body['repository']
    # Add your own GitHub Token to run it local
    token = os.environ.get(
        'GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
    GITHUB_URL = f"https://api.github.com/"
    headers = {
        "Authorization": f'token {token}'
    }
    params = {
        "state": "open"
    }
    repository_url = GITHUB_URL + "repos/" + repo_name
    # Fetch GitHub data from GitHub API
    repository = requests.get(repository_url, headers=headers)
    # Convert the data obtained from GitHub API to JSON format
    repository = repository.json()

    today = date.today()

    issues_reponse = []
    # Iterating to get issues for every month for the past 12 months
    for i in range(12):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        types = 'type:issue'
        repo = 'repo:' + repo_name
        ranges = 'created:' + str(last_month) + '..' + str(today)
        # By default GitHub API returns only 30 results per page
        # The maximum number of results per page is 100
        # For more info, visit https://docs.github.com/en/rest/reference/repos 
        per_page = 'per_page=100'
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = types + ' ' + repo + ' ' + ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "search/issues?q=" + search_query + "&" + per_page
        # requsets.get will fetch requested query_url from the GitHub API
        search_issues = requests.get(query_url, headers=headers, params=params)
        # Convert the data obtained from GitHub API to JSON format
        search_issues = search_issues.json()
        issues_items = []
        try:
            # Extract "items" from search issues
            issues_items = search_issues.get("items")
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        # if issues_items is None:
        #     continue
        for issue in issues_items:
            label_name = []
            data = {}
            current_issue = issue
            # Get issue number
            data['issue_number'] = current_issue["number"]
            # Get created date of issue
            data['created_at'] = current_issue["created_at"][0:10]
            if current_issue["closed_at"] == None:
                data['closed_at'] = current_issue["closed_at"]
            else:
                # Get closed date of issue
                data['closed_at'] = current_issue["closed_at"][0:10]
            for label in current_issue["labels"]:
                # Get label name of issue
                label_name.append(label["name"])
            data['labels'] = label_name
            # It gives state of issue like closed or open
            data['State'] = current_issue["state"]
            # Get Author of issue
            data['Author'] = current_issue["user"]["login"]
            issues_reponse.append(data)

        today = last_month

    df = pd.DataFrame(issues_reponse)

    # Daily Created Issues
    df_created_at = df.groupby(['created_at'], as_index=False).count()
    dataFrameCreated = df_created_at[['created_at', 'issue_number']]
    dataFrameCreated.columns = ['date', 'count']
    temp2 = dataFrameCreated.values.tolist()
    # issues_created_df1 =[]
    issues_created_df2 = []
    i=0
    while i < len(temp2):
        data = {}
        data['date'] = temp2[i][0]
        data['count'] = temp2[i][1]
        i=i+1
        arr = [data['date'],data['count']]
        issues_created_df2.append(arr)
        # issues_created_df1.append(data)



    '''
    Monthly Created Issues
    Format the data by grouping the data by month
    ''' 
    created_at = df['created_at']
    month_issue_created = pd.to_datetime(
        pd.Series(created_at), format='%Y-%m-%d')
    month_issue_created.index = month_issue_created.dt.to_period('m')
    month_issue_created = month_issue_created.groupby(level=0).size()
    month_issue_created = month_issue_created.reindex(pd.period_range(
        month_issue_created.index.min(), month_issue_created.index.max(), freq='m'), fill_value=0)
    month_issue_created_dict = month_issue_created.to_dict()
    created_at_issues = []
    for key in month_issue_created_dict.keys():
        array = [str(key), month_issue_created_dict[key]]
        created_at_issues.append(array)

    '''
    Monthly Closed Issues
    Format the data by grouping the data by month
    ''' 
    
    closed_at = df['closed_at'].sort_values(ascending=True)
    month_issue_closed = pd.to_datetime(
        pd.Series(closed_at), format='%Y-%m-%d')
    month_issue_closed.index = month_issue_closed.dt.to_period('m')
    month_issue_closed = month_issue_closed.groupby(level=0).size()
    month_issue_closed = month_issue_closed.reindex(pd.period_range(
        month_issue_closed.index.min(), month_issue_closed.index.max(), freq='m'), fill_value=0)
    month_issue_closed_dict = month_issue_closed.to_dict()
    closed_at_issues = []
    for key in month_issue_closed_dict.keys():
        array = [str(key), month_issue_closed_dict[key]]
        closed_at_issues.append(array)

    '''
        1. Hit LSTM Microservice by passing issues_response as body
        2. LSTM Microservice will give a list of string containing image paths hosted on google cloud storage
        3. On recieving a valid response from LSTM Microservice, append the above json_response with the response from
            LSTM microservice
    '''
    created_at_body = {
        "issues": issues_reponse,
        "type": "created_at",
        "repo": repo_name.split("/")[1]
    }
    closed_at_body = {
        "issues": issues_reponse,
        "type": "closed_at",
        "repo": repo_name.split("/")[1]
    }

    # Update your Google cloud deployed LSTM app URL (NOTE: DO NOT REMOVE "/")
    LSTM_API_URL = "https://lstm-oz2hqwmiuq-uc.a.run.app/" + "api/forecast"

    '''
    Trigger the LSTM microservice to forecasted the created issues
    The request body consists of created issues obtained from GitHub API in JSON format
    The response body consists of Google cloud storage path of the images generated by LSTM microservice
    '''
    created_at_response = requests.post(LSTM_API_URL,
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})
    
    '''
    Trigger the LSTM microservice to forecasted the closed issues
    The request body consists of closed issues obtained from GitHub API in JSON format
    The response body consists of Google cloud storage path of the images generated by LSTM microservice
    '''    
    closed_at_response = requests.post(LSTM_API_URL,
                                       json=closed_at_body,
                                       headers={'content-type': 'application/json'})

                
    #Tensorflow
    #print("CC:", created_at_body)
    
    created_at_response = requests.post(LSTM_API_URL,
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})


    pull_request_rpnse = requests.post(LSTM_API_URL_FINAL+"api/pulls",
                                       json=pull_request_body,
                                       headers={'content-type': 'application/json'})
    print("pull req res: ",pull_request_rpnse.json())
   
    commits_respnse = requests.post(LSTM_API_URL_FINAL+"api/commits",
                                       json=commits_body,
                                       headers={'content-type': 'application/json'})
    print("commits req res: ",commits_respnse.json())
    
    closed_at_response = requests.post(LSTM_API_URL,
                                       json=closed_at_body,
                                       headers={'content-type': 'application/json'})
    
    #stats model
    created_at_response_stat = requests.post(LSTM_API_URL_FINAL+"api/statmis",
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})

    print("create req stat res: ",created_at_response_stat.json())

    pull_request_rpnse_stat = requests.post(LSTM_API_URL_FINAL+"api/statmpull",
                                       json=pull_request_body,
                                       headers={'content-type': 'application/json'})

    print("pull req stat res: ",pull_request_rpnse_stat.json())
   
    commits_respnse_stat = requests.post(LSTM_API_URL_FINAL+"api/statmcommits",
                                       json=commits_body,
                                       headers={'content-type': 'application/json'})

    print("commits req stat res: ",commits_respnse_stat.json())
    
    closed_at_response_stat = requests.post(LSTM_API_URL_FINAL+"api/statmisc",
                                       json=closed_at_body,
                                       headers={'content-type': 'application/json'})

    print("create req stat res: ",closed_at_response_stat.json())

    
    #fb profet api routes
    created_at_response_fb = requests.post(LSTM_API_URL_FINAL+"api/fbprophetis",
                                        json=created_at_body,
                                        headers={'content-type': 'application/json'})

    print("create req fb res: ",created_at_response_fb.json())

    pull_request_fb_rpnse = requests.post(LSTM_API_URL_FINAL+"api/fbprophetpull",
                                       json=pull_request_body,
                                       headers={'content-type': 'application/json'})

    print("pull req fb res: ",pull_request_fb_rpnse.json())
   
    commits_fb_respnse = requests.post(LSTM_API_URL_FINAL+"api/fbprophetcommits",
                                       json=commits_body,
                                       headers={'content-type': 'application/json'})

    print("commits req res: ",commits_fb_respnse.json())
    
    closed_at_fb_response = requests.post(LSTM_API_URL_FINAL+"api/fbprophetisc",
                                       json=closed_at_body,
                                       headers={'content-type': 'application/json'})

    print("create req fb res: ",closed_at_fb_response.json())

    # For Stars & fork 
    total_count = []
    fork_count = []
    i=0
    for repos in repositories: 
        query_url = GITHUB_URL + "repos/" + repos
        # requsets.get will fetch requested query_url from the GitHub API
        resp = requests.get(query_url, headers=headers, params=params)
        resp = resp.json()
        arr = [names[i],resp.get("stargazers_count")]
        arr2 = [names[i],resp.get("forks_count")]
        total_count.append(arr)
        fork_count.append(arr2)
        i=i+1
    stars_resp = total_count
    fork_resp = fork_count


# For the first 3 in the lstm 
    today = date.today()
    last_year = today + dateutil.relativedelta.relativedelta(years=-2)
    # To get issues for all repo
    total_count = []
    i=0
    for repos in repositories: 
        types = 'type:issue'
        repo = 'repo:' + repos
        ranges = 'created:' + str(last_year) + '..' + str(today)
        search_query = types + ' ' + repo + ' ' + ranges
        query_url = GITHUB_URL + "search/issues?q=" + search_query
        # requsets.get will fetch requested query_url from the GitHub API
        resp = requests.get(query_url, headers=headers, params=params)
        resp = resp.json()
        arr = [names[i],resp.get("total_count")]
        total_count.append(arr)
        i=i+1
    issues_resp = total_count

    df_temp = df.groupby(['created_at']).count().reset_index()
    data = df_temp.sort_values('issue_number', ascending=False).head(1).iloc[0]['created_at']
    max_issues_day = data + " " + convert_to_day(data)
    df_temp = df.groupby(['closed_at']).count().reset_index()
    data = df_temp.sort_values('issue_number', ascending=False).head(1).iloc[0]['closed_at']
    max_close_day = data + " " + convert_to_day(data)
    month = df
    month['closed_month_year'] = pd.to_datetime(df['closed_at']).dt.to_period('M')
    max_issue_month = month.groupby('closed_month_year').count().idxmax(axis=0, skipna = True)
    max_issue_month = max_issue_month['created_at'].strftime('%B %F')

    issues_reponse = []
    types = 'type:pr'
    repo : 'repo:' + repo_name
    ranges = 'created:' + str(last_year) + '..' + str(today)
    search_query = types + ' ' + repo + ' ' + ranges
    query_url = GITHUB_URL + "search/issues?q=" + search_query
    # requsets.get will fetch requested query_url from the GitHub API
    resp = requests.get(query_url, headers=headers, params=params)
    search_issues = resp.json()
    issues_items = []
    try:
        # Extract "items" from search issues
        issues_items = search_issues.get("items")
    except KeyError:
        error = {"error": "Data Not Available"}
        resp = Response(json.dumps(error), mimetype='application/json')
        resp.status_code = 500
        return resp
    for issue in issues_items:
        label_name=[]
        data = {}
        current_issue = issue
        # Get issue number
        data['issue_number'] = current_issue["number"]
        # Get created date of issue
        data['created_at'] = current_issue["created_at"][0:10]
    #    if current_issue["closed_at"] == None: data['closed_at']= current_issue["closed_at"]
    #     else:
    #         data['closed_at']= current_issue["closed_at"][0:10]               # Get closed date of issue
        for label in current_issue["labels"]:
            label_name.append(label["name"])                                  # Get label name of issue
        data['labels']= label_name
        data['State'] = current_issue["state"]                                # It gives state of issue like closed or open
        data['Author'] = current_issue["user"]["login"] 
        issues_reponse.append(data)

    df2 = pd.DataFrame(issues_reponse)

    df_pr_at = df2.groupby(['created_at'], as_index=False).count()
    dataFramePullReq = df_pr_at[['created_at', 'issue_number']]
    dataFramePullReq.columns = ['date', 'count']
    temp3 = dataFramePullReq.values.tolist()
    issues_pr_df3 =[]
    i=0
    while i < len(temp3):
        data = {}
        data['date'] = temp3[i][0]
        data['count'] = temp3[i][1]
        i=i+1
        issues_pr_df3.append(data)



    pull_at_body = {
        "issues": issues_reponse,
        "type": "pull_issues"
    }

    created_at_pulls_response = requests.post(LSTM_API_URL,
                                        json=pull_at_body,
                                        headers={'content-type': 'application/json'})
    # build
    query_url = GITHUB_URL + "users/" + repo_name.split("/")[0] + "/repos" 
    page_no = 1
    repos_data = []
    count=0
    while (True):
        response = requests.get(query_url, headers=headers, params=params)
        response = response.json()
        repos_data = repos_data + response
        repos_fetched = len(response)
        if (repos_fetched == 30 and count <100):
            page_no = page_no + 1
            url = query_url + '?page=' + str(page_no)
            count = count + 1
        else:
            break
    repo_req = []
    k=0
    while k < len(repos_data):
        url = GITHUB_URL + "repos/" + repo_name
        if(repos_data[k]['url'] == url) :
            repo_req.append(repos_data[k])
        k = k+1
    repos_info = []
    for i, repo in enumerate(repo_req):
        data = []
        data.append(repo['id'])
        data.append(repo['name'])
        data.append(repo['description'])
        data.append(repo['created_at'])
        data.append(repo['updated_at'])
        data.append(repo['owner']['login'])
        data.append(repo['license']['name'] if repo['license'] != None else None)
        data.append(repo['has_wiki'])
        data.append(repo['forks_count'])
        data.append(repo['open_issues_count'])
        data.append(repo['stargazers_count'])
        data.append(repo['watchers_count'])
        data.append(repo['url'])
        data.append(repo['commits_url'].split("{")[0])
        data.append(repo['url'] + '/branches?')
        data.append(repo['url'] + '/contributors')
        data.append(repo['releases_url'])
        data.append(repo['url'] + '/languages')
        repos_info.append(data)
    df = pd.DataFrame(repos_info,columns = ['Id', 'Name', 'Description', 'created_at', 'updated_at', 
                                                        'Owner', 'License', 'has_wiki', 'forks_count', 
                                                        'open_issues_count', 'stargazers_count', 'watchers_count',
                                                        'url', 'commits_url','branches_url','contributors_url','releases_url', 'languages_url'])

    '''
    Create the final response that consists of:
        1. GitHub repository data obtained from GitHub API
        2. Google cloud image urls of created and closed issues obtained from LSTM microservice
    '''
    """
    Creates a JSON response with data about GitHub repository issues, pull requests, commits, etc.

    Args:
    - created_at_issues: List of issues created.
    - week_created_at_issues: List of issues created weekly.
    - closed_at_issues: List of issues closed.
    - week_closed_at_issues: List of issues closed weekly.
    - repository: Dictionary with repository data such as star and fork count.
    - *_response_stat, *_response_fb, *_response: Response objects for various metrics.

    Returns:
    - JSON response with the aggregated data.
    """

    # Construct the JSON response
    json_response = {
        # Basic repository statistics
        "created": created_at_issues,
        "created_weekly": week_created_at_issues,
        "closed": closed_at_issues,
        "closed_weekly": week_closed_at_issues,
        "starCount": repository["stargazers_count"],
        "forkCount": repository["forks_count"],

        # URLs for statistical data visualizations
        "stat_createdAtImageUrls": created_at_response_stat.json(),
        "stat_closedAtImageUrls": closed_at_response_stat.json(),
        "stat_pullReqImageUrls": pull_request_rpnse_stat.json(),
        "stat_commitsImageUrls": commits_respnse_stat.json(),

        # URLs for Facebook-based data visualizations
        "fb_createdAtImageUrls": created_at_response_fb.json(),
        "fb_closedAtImageUrls": closed_at_fb_response.json(),
        "fb_pullReqImageUrls": pull_request_fb_rpnse.json(),
        "fb_commitsImageUrls": commits_fb_respnse.json(),

        # URLs for general data visualizations
        "createdAtImageUrls": created_at_response.json(),
        "closedAtImageUrls": closed_at_response.json(),
        "pullReqImageUrls": pull_request_rpnse.json(),
        "commitsImageUrls": commits_respnse.json(),
    }
    # Return the response back to client (React app)
    return jsonify(json_response)


# Run flask app server on port 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
