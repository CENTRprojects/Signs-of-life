import dns
import dns.rdatatype
import dns.resolver as reso

from memory_guard import adaptive_parallel
import pandas as pd

from config import RUN_CONFIG

def extract_dns_data(documents, datatype):

    # features
    X_first = select_features_dns(documents, datatype)

    # First classification
    if RUN_CONFIG["MULTI_PROCESSING"]:
        list_res = adaptive_parallel(extract_dns, X_first, phase="dns_" + datatype)
    else:
        # non parallel
        list_res = []
        for doc in X_first:
            list_res.append(extract_dns(doc))
    return pd.DataFrame(list_res)

def select_features_dns(documents, datatype):
    return [{"url": doc.url, "datatype": datatype} for doc in documents]

def extract_dns(feats):
    url = feats["url"]
    datatype = feats["datatype"]
    has_datatype = "dns_has_" + datatype

    response = ""
    comment = ""
    resolver = reso.Resolver()
    resolver.use_edns(0,dns.flags.DO,4096)
    resolver.nameservers = ([RUN_CONFIG["RESOLVER_NAMESERVER"]])
    resolver.timeout = 2.0
    resolver.lifetime = 5.0
    rdtype = dns.rdatatype.DNSKEY
    rdclass = dns.rdataclass.IN

    #dmarc is a special TXT request, and is not available as a dns.rdatatype
    if(datatype == "dmarc"):
        dmarc_domain = f"_dmarc.{url}"
        try:
            answers = resolver.resolve(dmarc_domain, 'TXT')
            for rdata in answers:
                response = rdata.to_text()
        except Exception as e:
            comment = f"Error fetching DMARC record: {e}"
    elif(datatype == "DKIM"):
        try:
            ns = extract_dns({"url": url, "datatype": 'NS'})['dns_value_NS'].split()[0][:-1]
            ip = extract_dns({"url": ns, "datatype": 'A'})['dns_value_A'].split()[0]
            qname = dns.name.from_text(f"_domainkey.{url}")
            q = dns.message.make_query(qname, dns.rdatatype.SRV)
            r = dns.query.udp(q, ip, timeout = RUN_CONFIG['DNS_DKIM_TIMEOUT'])
            var_rcode = r.rcode()
            if(dns.rcode.Rcode.to_text(var_rcode)=='NOERROR'):
                response = 'NOERROR answer received. This implies DKIM exists for this domain'
            elif(dns.rcode.Rcode.to_text(var_rcode)=='NXDOMAIN'):
                comment = 'NXDOMAIN answer received. This implies no DKIM exists for this domain'
            else:
                comment = 'An unexpected response'
        except IndexError as e:
            comment = 'Domain NS not found, unable to check for DKIM'
        except Exception as e:
            comment = f"Error fetching DKIM record: {e}"

    elif(datatype == "MX"):
        try:
            answers = reso.resolve(url, 'MX')  # system resolver
            parts = []
            for rdata in answers:
                parts.append(f"{int(rdata.preference)} {rdata.exchange}")
            response = " | ".join(parts)
        except reso.NoAnswer:
            comment = "No answer"
        except reso.NoNameservers:
            comment = "No name server"
        except dns.exception.Timeout:
            comment = "Timeout"
        except reso.NXDOMAIN:
            comment = "No existing query name"
        except Exception as e:
            print("Special error: {} - {}".format(type(e), str(e)))
            comment = str(e)


    else:
        # Making the function dynamic for any availabe dns.rdatatype classes
        try:
            if datatype == 'RRSIG':
                dns_class = getattr(dns.rdatatype, datatype)
                records = resolver.resolve(url, rdtype, rdclass, True).response
                response = records.find_rrset(records.answer, url, rdclass, dns_class, rdtype)
            else:
                records = resolver.resolve(url, datatype, rdclass, True)
                try:
                    for server in records:
                        response += str(server.target) + " "
                except:
                    for server in records:
                        response += str(server) + " "
        except reso.NoAnswer as e:
            # except dns.resolver.NoAnswer:
            comment = "No answer"
        except reso.NoNameservers as e:
            # except dns.resolver.NoNameservers as e:
            comment = "No name server"
        except dns.exception.Timeout:
            comment = "Timeout"
        except reso.NXDOMAIN:
            comment = "No existing query name"
        except Exception as e:
            print("Special error: {} - {}".format(type(e), str(e)))
            comment = str(e)
    if response == "":
        has_dns_data = False
    else:
        has_dns_data = True

    rs = {"url": url, has_datatype: has_dns_data, "dns_value_" + datatype: response, "dns_" + datatype + "_comment": comment}

    return rs



if __name__ == '__main__':
    print(extract_dns({"url": 'google.com', "datatype": 'RRSIG'}))
    print(extract_dns({"url": 'google.com', "datatype": 'A'}))
    print(extract_dns({"url": 'google.com', "datatype": 'NS'}))
    print(extract_dns({"url": 'google.com', "datatype": 'TXT'}))
    print(extract_dns({"url": 'google.com', "datatype": 'AAAA'}))
    print(extract_dns({"url": 'google.com', "datatype": 'SOA'}))
    print(extract_dns({"url": 'google.com', "datatype": 'CAA'}))
    print(extract_dns({"url": 'example.com', "datatype": 'CNAME'}))
    print(extract_dns({"url": 'example.com', "datatype": 'DKIM'}))