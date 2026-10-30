## Source: fleurs

| Variant | Files | WER | CER | Script% | RTF |
|---|---|---|---|---|---|
| base | 12 | 78.9% | 69.6% | 0.0% | 0.51 |
| base+prompt | 12 | 79.3% | 56.8% | 88.1% | 1.74 |
| small | 12 | 54.0% | 39.1% | 81.1% | 2.70 |
| small+prompt | 12 | 51.3% | 36.9% | 100.0% | 3.27 |

WER by language (files in parentheses):

| Variant | en | hi | te |
|---|---|---|---|
| base | 2.8% (4) | 100.0% (4) | 122.6% (4) |
| base+prompt | 2.8% (4) | 91.4% (4) | 141.9% (4) |
| small | 5.6% (4) | 54.7% (4) | 108.1% (4) |
| small+prompt | 5.6% (4) | 53.1% (4) | 100.0% (4) |

Script% by language (files in parentheses):

| Variant | en | hi | te |
|---|---|---|---|
| base | n/a (4) | 0.0% (4) | 0.0% (4) |
| base+prompt | n/a (4) | 100.0% (4) | 76.1% (4) |
| small | n/a (4) | 100.0% (4) | 26.9% (4) |
| small+prompt | n/a (4) | 100.0% (4) | 100.0% (4) |

Warnings:

- base, hi: Script% 0.0% (< 50%) and WER 100.0% (> 80%) over 4 file(s). WER is dominated by script mismatch, not recognition quality.
- base, te: Script% 0.0% (< 50%) and WER 122.6% (> 80%) over 4 file(s). WER is dominated by script mismatch, not recognition quality.
- small, te: Script% 26.9% (< 50%) and WER 108.1% (> 80%) over 4 file(s). WER is dominated by script mismatch, not recognition quality.

Hardware: Windows-11-10.0.26200-SP0; AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD; 8 logical CPUs. Device: cpu; compute type: int8. RTF values are wall-clock on this machine; timings from the Render free-tier deployment are not comparable to laptop timings.
