"""
从 pcap 文件中提取 DNS 域名
"""
import os
from scapy.all import rdpcap, DNS, DNSQR


def extract_domains_from_pcap(pcap_path):
    """
    从 pcap 文件中提取所有 DNS 查询域名
    
    Args:
        pcap_path: pcap 文件路径
        
    Returns:
        list: 域名列表
    """
    domains = []
    try:
        packets = rdpcap(pcap_path)
        for packet in packets:
            if packet.haslayer(DNS) and packet.haslayer(DNSQR):
                dns_layer = packet[DNS]
                if dns_layer.qr == 0:  # 查询包
                    query_name = dns_layer.qd.qname.decode('utf-8').rstrip('.')
                    if query_name:
                        domains.append(query_name)
    except Exception as e:
        print(f"处理文件 {pcap_path} 时出错: {e}")
    
    return domains


def extract_all_domains(dataset_dir, label):
    """
    从数据集目录中提取所有域名
    
    Args:
        dataset_dir: 数据集目录路径
        label: 标签 (0=正常, 1=恶意)
        
    Returns:
        list: (域名, 标签) 元组列表
    """
    all_data = []
    
    # 递归遍历目录
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.pcap', '.pcapng')):
                file_path = os.path.join(root, file)
                domains = extract_domains_from_pcap(file_path)
                for domain in domains:
                    all_data.append((domain, label))
                print(f"从 {file_path} 提取了 {len(domains)} 个域名")
    
    return all_data

