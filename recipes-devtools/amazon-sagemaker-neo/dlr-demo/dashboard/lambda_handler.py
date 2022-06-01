####################################
#
#    Copyright 2022 NXP
#
####################################


import boto3
import json

client = boto3.client('iot', region_name="us-east-2")

def lambda_handler(event, context):
    response = client.list_things(
        #nextToken='string',
        maxResults=123,
        #attributeName='',
        #attributeValue='string',
        #thingTypeName='string',
        usePrefixAttributeValue=False
        )
    # TODO implement
    things = []
    for thing in response['things']:
        try:
            thing['lat'] = float(thing['attributes']['lat'] )
            thing['lng'] = float(thing['attributes']['lng'] )
        except:
            thing['lat'] = 0
            thing['lng'] = 0
        thing.pop('attributes')
        things.append(thing)
    return {
        'statusCode': 200,
        'things': things
    }
