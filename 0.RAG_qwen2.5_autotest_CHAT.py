# Chat with an intelligent assistant in your terminal  
# MODEL: https://huggingface.co/Qwen
# qwen2.5-1.5b-instruct-q5_k_m.gguf
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
import datetime
import random
import string
import tiktoken
import argparse

#Add GPU argument in the parser
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true")

args = parser.parse_args()
GPU = args.gpu
if GPU:
    ngpu_layers = 28
    print(f'Selected GPU: offloading {ngpu_layers} layers...')
else:
     ngpu_layers = 0   
     print('Loading Model on CPU only......')

prmpt_tasks = ["introduction",
               "explain in one sentence",
               "explain in three paragrapghs",
               "summarize",
               "Summarize in two sentences",
               "Write in a list the three main key points",
               "Table of Contents",
               "RAG",
               "Truthful RAG",
               "write content from a reference",
               "extract 5 topics",
               "Creativity: 1000 words SF story",
               "Reflection prompt"
               ]
prmpt_coll = [
"""Hi there my name is Fabio. who are you?""",
"""explain in one sentence what is science.\n""",
"""explain in three paragraphs what is artificial intelligence.\n""",
"""summarize the following text:

[text]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of text]

""",
"""Summarize in two sentences the following text

[text]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of text]

""",
"""Write in a list the three main key points of the following text:

[text]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of text]

""",
"""A "table of content" is an ordered list of the topic contained in the text: write the "Table of Contents" of the following text. 

[text]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of text]

 """,
"""Reply to the question only using the provided context. If the answer is not contained in the text say "unanswerable".

question: what Chine achieved with it's long-term planning?

[context]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of context]

answer:
 """,
"""Reply to the question only using the provided context. If the answer is not contained in the text say "unanswerable".

question: who is Anne Frank?

[context]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of context]
Remember: if you cannot answer based on the provided context, say "unanswerable"

answer:
 """, 

"""Using the following text as a reference, write a 5-paragraphs essay about "the benefits of China economic model".

[text]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of text]

""",
"""write the five most important topics from the following text:

[text]One of the things everybody in the West knows about China is that it is not a democracy, and is instead a regime run with an iron fist by a single entity, the Chinese Communist Party, whose leadership rarely acts transparently, running the country without the need for primary elections, alternative candidacies, etc.
In general, those of us who live in democracies, with relatively transparent electoral processes, tend to consider the Chinese system undesirable, little more than a dictatorship where people have no say in who governs them.
That said, among the “advantages” of the Chinese system is that because the leadership never has to put its legitimacy to the vote, it can carry out very long-term planning in the knowledge that another administration isn’t going to come along and change those plans.
Obviously, I put “advantages” in quotation marks because, as democrats, most of my readers would never be willing to sacrifice their freedom for greater planning, but there is no doubt that China, since its system works like this and its population seems to have accepted it for generations, intends to turn this into a comparative advantage, the term used in business when analyzing companies.
It turns out that China’s capacity for long-term planning is achieving something unheard of in the West: it seems the country reached peak carbon dioxide and greenhouse gas emissions in 2023, and that the figures for 2024, driven above all by a determined increase in the installation of renewable energies, are not only lower, but apparently going to mark a turning point.
If you don’t believe what you’re reading, rub your eyes: we’re talking about China, the country with the highest emissions in the world, which claimed that its arrival at Net Zero would be delayed by a few decades because the country had the right to consolidate its economy by burning the fossil fuels it deemed necessary, in the same way that the West had done for the last century or more.
India makes the same argument: the economies that we consider developed today have spent decades emitting carbon dioxide like there were no tomorrow, therefore, they should have the right to do so at least until their economies have reached a similar level of development.
Given that we are talking about the two most populous economies in the world, these arguments are problematic, given that we all live on the same planet, and it remains to be seen whether our species can continue to inhabit it if emissions continue to grow. No joke. And if you are somebody who denies climate change, stop reading this article now and please don’t expose yourself to ridicule in the comments section, instead do yourself the favor of reading a little before coming back here.
What is happening in China? Well, in addition to having established technological leadership in solar panels and batteries, the two most strategic technologies for decarbonization and the energy transition, it has decided to fully commit to them and deploy them at a much higher speed than initially planned. Why? For the simplest reason of all: they are much cheaper.
While we in the West still complain that EVs are more expensive, or argue if they really do reduce emissions, or if batteries can be recycled, in China they are no longer the future, but the present, while solar panels and wind turbines are put everywhere they can reasonably be placed, with batteries installed to cover intermittency.
The result is that the world’s biggest polluter may have peaked in emissions in 2023, and already be in the downward phase. The expansion of solar and wind generation meant that by March 2024 these sources covered 90% of the growth in electricity demand. Together with a very strong commitment to hydroelectric power, with some of the largest dams in the world, we are facing a commitment that will not only ensure all the country’s energy needs, but do so at significantly lower costs.
It makes perfect sense that China is now the largest exporter of EVs: it is the logical evolution of a long-term planned economy based on an understanding of the cost advantages that technology provides. Sure, it is hedging its bets by planning more coal or nuclear power plants, but they are no longer the first option.
China and India were until recently the planet’s biggest polluters, but they now offer a model for energy transition (there is still a long way to go; but we are talking about models, not a done deal).
It could soon be the case that the so-called developing countries will be showing the West the way forward.
[end of text]

""",
"""Science Fiction: The Last Transmission - Write a story that takes place entirely within a spaceship's cockpit as the sole surviving crew member attempts to send a final message back to Earth before the ship's power runs out. The story should explore themes of isolation, sacrifice, and the importance of human connection in the face of adversity. 800-1000 words.

""",
"""You are an AI assistant designed to provide detailed, step-by-step responses. Your outputs should follow this structure:
1. Begin with a <thinking> section.
2. Inside the thinking section:
   a. Briefly analyze the question and outline your approach.
   b. Present a clear plan of steps to solve the problem.
   c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
3. Include a <reflection> section for each idea where you:
   a. Review your reasoning.
   b. Check for potential errors or oversights.
   c. Confirm or adjust your conclusion if necessary.
4. Be sure to close all reflection sections.
5. Close the thinking section with </thinking>.
6. Provide your final answer in an <output> section.
Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. Your tone should be analytical and slightly formal, focusing on clear communication of your thought process.
Remember: Both <thinking> and <reflection> MUST be tags and must be closed at their conclusion
Make sure all <tags> are on separate lines with no other text. Do not include other text on a line containing a tag.

user question: explain why it is crucial for teachers to learn how to use generative AI for their job and for the future of education. Include relevant learning path for teachers and educators. 

"""
]

#print(len(prmpt_coll))  #11


encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))
modelname = 'qwen2.5-1.5b-instruct'
def countTokens(text):
    encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))
    numoftokens = len(encoding.encode(text))
    return numoftokens

def writehistory(filename,text):
    # save the user/assistant pairs into a logfile located in the logs subdirectory
    with open(f'{filename}', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

# create THE LOG FILE 
logfile = f'QWEN2.5-1.5b_{genRANstring(5)}_log.txt'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 💻 {modelname}\n---\n🧠🫡: You are a helpful assistant.')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')

print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
from llama_cpp import Llama
llm = Llama(
            model_path='qwen2.5-1.5b-instruct-q5_k_m.gguf',
            n_gpu_layers=ngpu_layers,
            temperature=1.1,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=1500,
            repeat_penalty=1.178,
            stop=['<|im_end|>'],
            verbose=False,
            )
print("2. Model qwen2.5-1.5b-instruct-q5_k_m.gguf loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = []
print("\033[92;1m")
print(f'📝Logfile: {logfilename}')

############################# AUTOMATIC PROMPTING EVALUATION  11 TURNS #################################
for i in range(0,len(prmpt_coll)):
    test = []
    print(f'NLP TAKS>>> {prmpt_tasks[i]}')
    print("\033[91;1m")  #red
    print(prmpt_coll[i])
    test.append({"role": "user", "content": prmpt_coll[i]})
    print("\033[92;1m")
    # Preparing Generation history pair
    new_message = {"role": "assistant", "content": ""}
    full_response = ""
    fisrtround = 0
    start = datetime.datetime.now()
    print("💻 > ", end="", flush=True)
    for chunk in llm.create_chat_completion(
        messages=test,
        temperature=1.2,
        repeat_penalty= 1.178,
        stop=['<|im_end|>'],
        max_tokens=1500,
        stream=True,):
        try:
            if chunk["choices"][0]["delta"]["content"]:
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                full_response += chunk["choices"][0]["delta"]["content"]                              
        except:
            pass        
    new_message["content"] = full_response
    test.append(new_message)   
    delta = datetime.datetime.now() - start
    totalseconds = delta.total_seconds()
    output = full_response
    prompttokens = countTokens(prmpt_coll[i])
    assistanttokens = countTokens(output)
    totaltokens = prompttokens + assistanttokens
    speed = totaltokens/totalseconds
    genspeed = assistanttokens/totalseconds
    print('')
    print("\033[91;1m")
    rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
    print("\033[92;1m")
    stats = f'''---
Prompt Tokens: {prompttokens}
Output Tokens: {assistanttokens}
TOTAL Tokens: {totaltokens}
>>>⏱️Inference time: {delta}
>>>🧮Inference speed: {speed:.3f}  t/s
>>>🏃‍♂️Generation speed: {genspeed:.3f}  t/s
📝Logfile: {logfilename}
>>>💚User rating: {rating}

'''
    print(stats)
    writehistory(logfilename,f'''✅NLP TAKS>>> {prmpt_tasks[i]}
👨‍💻: {prmpt_coll[i]}
💻 > {full_response}
{stats}
📝Logfile: {logfilename}''')


############################# START TURN BASED CHAT #################################
print('Starting now Normal Chat turn based interface...')
counter = 1
while True:
    # Reset history after 3 turns
    if counter > 5:
        history = []        
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    history.append({"role": "user", "content": userinput})
    print("\033[92;1m")
    # Preparing Generation history pair
    new_message = {"role": "assistant", "content": ""}
    # Starting generation loop
    full_response = ""
    fisrtround = 0
    start = datetime.datetime.now()
    print("💻 > ", end="", flush=True)
    for chunk in llm.create_chat_completion(
        messages=history,
        temperature=1.2,
        repeat_penalty= 1.178,
        stop=['<|im_end|>'],
        max_tokens=1500,
        stream=True,):
        try:
            if chunk["choices"][0]["delta"]["content"]:
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                full_response += chunk["choices"][0]["delta"]["content"]                              
        except:
            pass        
    new_message["content"] = full_response
    history.append(new_message)  
    counter += 1  
    delta = datetime.datetime.now() - start
    totalseconds = delta.total_seconds()
    output = full_response
    prompttokens = countTokens(userinput)
    assistanttokens = countTokens(output)
    totaltokens = prompttokens + assistanttokens
    speed = totaltokens/totalseconds
    genspeed = assistanttokens/totalseconds
    print('')
    print("\033[91;1m")
    rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
    print("\033[92;1m")
    stats = f'''---
Prompt Tokens: {prompttokens}
Output Tokens: {assistanttokens}
TOTAL Tokens: {totaltokens}
>>>⏱️Inference time: {delta}
>>>🧮Inference speed: {speed:.3f}  t/s
>>>🏃‍♂️Generation speed: {genspeed:.3f}  t/s
📝Logfile: {logfilename}
>>>💚User rating: {rating}

'''
    print(stats)
    writehistory(logfilename,f'''👨‍💻: {userinput}
💻 > {full_response}
{stats}
📝Logfile: {logfilename}''')


