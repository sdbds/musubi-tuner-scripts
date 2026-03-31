import os
base = r'c:\Users\17290\Desktop\BaiduSyncdisk\SD\musubi-tuner-scripts\qinglong-captions\datasets'
total = 0
bad = 0
for r, d, fs in os.walk(base):
    for f in sorted(fs):
        if f.endswith('.txt'):
            total += 1
            path = os.path.join(r, f)
            with open(path, 'r', encoding='utf-8') as fh:
                c = fh.read()
            nl = c.count('\n')
            bs = c.count('\b')
            if nl > 0 or bs > 0:
                bad += 1
                print('BAD: ' + f + ' (newlines=' + str(nl) + ' backspaces=' + str(bs) + ')')
            else:
                print('OK: ' + f + ' (' + str(len(c)) + ' chars)')
print('\nTotal: ' + str(total) + ', Issues: ' + str(bad))
