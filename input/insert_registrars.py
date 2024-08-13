#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, csv, re
from collections import namedtuple

def PrintInstructions():
	print("""
	This file will insert registrars into the hosting_companies_with_tld.csv file for use with the crawler.

	To use this script, first make a file (or files) with a list of registrar domains with one on each line. 
	As long as the domain is there in some proper form it will be extracted at the third level - (domain)(.subtld.tld):

	file1.txt:
	--------------------------------
	registrar.com
	registrar.co.uk
	registrar.xyz
	--------------------------------

	file2.csv:
	--------------------------------
	"registrar2.pl"
	www.registrar3.com # will extract to registrar3.com
	https://registrar4.com.au
	--------------------------------
			
	Then run:

	python insert_registrars.py file1.txt file2.csv

	Optional flags:
	-rr			write Regex Record. This will create a file regex_record.csv, which records how each domain is captured (useful for debugging)

	""")

full_list = {}
old_list = set()
old_list_dict = {}
DOMAIN_REGEX = re.compile('(www\.)?([-a-z0-9]+(\.[-0-9a-z]+)?(\.[a-z]+))')
domains_found = set()
domain_counter = 0
all_domains_found = []
duplicates = {}
Record = namedtuple('Record', ['raw', 'regex'])
regex_record = []

def extract_domain(domain):
	domains = DOMAIN_REGEX.search(domain)
	if domains is None: 
		print(f"{domain} is not a domain")
		return None
	return domains.group(2)

def record_regex(domain):
	global regex_record
	found = extract_domain(domain)
	record = Record(domain.strip('\n'), found)
	regex_record.append(record)

def add(domain):
	""" check the incoming domain is a domain and then add it to the global dict """
	global domain_counter
	global domains_found
	global all_domains_found
	found = extract_domain(domain)
	if found:
		domain_counter += 1
		domains_found.add(found)
		all_domains_found.append(found)

def load_existing():
	#global full_list
	global old_list
	with open('hosting_companies_with_tld.csv','r') as f:
		csvreader = csv.reader(f, delimiter=',')
		for reg in csvreader:
			if csvreader.line_num == 1:
				continue
			old_list_dict[reg[0].lower()] = set(reg[1].lower().split(';'))
			domain = reg[0].lower()
			tlds = set(reg[1].lower().split(';'))
			for tld in tlds:
				old_list.add(domain+tld)

def calc_size(dictionary):
	counter = 0
	for domain in dictionary:
		counter += len(dictionary[domain])
	return counter

def write_full():
	with open('hosting_companies_with_tld.csv','w', newline='', encoding='utf-8') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['hosting_name','tld'])
		for domain in full_list:
			writer.writerow([domain,';'.join(full_list[domain])])
	
def write_registrars_by_tld():	
	# decompress the full_list dict into a set, sort that into tlds, then save those out
	reg_by_tld = {}
	for host in full_list:
		for post in full_list[host]:
			# create full domain then split by 1st level tld only
			domain = host+post
			tld = '.'+domain.split('.')[-1]
			if tld not in reg_by_tld:
				reg_by_tld[tld] = set()
			reg_by_tld[tld].add(domain)

	# write out the registrar by tld files
	dirname = 'registrars_by_tld'
	if not os.path.isdir(dirname):
		os.mkdir(dirname)
	for tld in reg_by_tld:
		filename = os.path.join(dirname, f"registrar_by_tld {tld}.txt")
		if os.path.exists(filename):
			os.remove(filename)
		with open(filename, 'w', newline='', encoding='utf-8') as f:
			for domain in reg_by_tld[tld]:
				f.write(domain+'\n')
	print(f"{len(reg_by_tld)} files written")
	
	return f'Converted {calc_size(full_list)} registrars to {calc_size(reg_by_tld)}'

def DoIt(filenames):
	global domains_found
	global all_domains_found
	global duplicates
	load_existing()
	original_size = calc_size(old_list_dict)
	domains_found = set()
	all_domains_found = []
	# cheap trick
	capture_regex = True if '-rr' in filenames else False
	for filename in filenames:
		if filename != '-rr':
			with open(filename, 'r') as f:
				for domain in f:
					add(domain.lower())
					if capture_regex:
						record_regex(domain)
				if len(all_domains_found) != len(domains_found):
					dupecheck = {}
					for domain in all_domains_found:
						if domain not in dupecheck:
							dupecheck[domain] = 1
						else:
							# print(f"Duplicate found: {domain}")
							if domain not in duplicates:
								duplicates[domain] = 1
							else:
								duplicates[domain] += 1

	unique_new_domains = len(domains_found)
	if unique_new_domains > 0:
		print("\nAdding these unique domains to the registrar file:")
		for domain in domains_found:
			print(domain)
	# add old list to domains_found
	for domain in old_list:
		add(domain.lower())
	# add domains found to full list
	domains_added = set()
	for domain in domains_found:
		try:
			pre,post = domain.split('.',1)
		except:
			print(f"but why no split? domain : {domain}")
		#print(f'{domain} = {pre}, {post}')
		post = '.'+post
		if pre not in full_list:
			full_list[pre] = set()
		if post not in full_list[pre]:
			domains_added.add(domain)
		full_list[pre].add(post)


	print('\n')
	if len(duplicates) > 0:
		print(f"Found {len(duplicates)} duplicates in given files (saved into duplicates.log)")
		with open('duplicates.log','w') as f:
			for key,value in duplicates.items():
				f.write(f'{key}\t{value}\n')
	# save full list out into original format
	write_full()

	# write out registrars_by_tld
	msg = write_registrars_by_tld()

	print(f'Found {unique_new_domains} distinct domains in provided files')
	print(f'Found {len(old_list)} distinct domains in old file, which originally had {original_size}')
	print(f'After adding everything in the provided files, we now have a total of {len(domains_found)} unique registrar domains')
	print(f'Added {calc_size(full_list) - original_size} new registrar urls to hosting file ({original_size} to {calc_size(full_list)})')
	print(msg)
	if len(domains_added) > 0 and len(domains_added) < 100:
		print(f'domains added: {domains_added}')
	print('\n')
	if capture_regex is True:
		with open('regex_record.csv', 'w', newline='', encoding='utf-8') as csvfile:
			print('Writing Regex record')
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(['raw', 'regex'])
			csv_writer.writerows(regex_record)
			print('Regex Record written\n')


if __name__ == "__main__":
	if len(sys.argv) > 1:
		print(f'\nInserting Registrars from: {" ".join(sys.argv[1:])}\n')
		DoIt(sys.argv[1:])
	else:
		PrintInstructions()