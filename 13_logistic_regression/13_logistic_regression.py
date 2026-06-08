# 한글을 지원하지 않음
# 환경제어 : matplotlibrc 파일, plt.rcParams, plt.rc()
# 현재 디렉토리, 설치된 디렉토리
import matplotlib.pyplot as plt # 시각화 script
# !apt-get update -qq # ubuntu 설치 명령어
# !apt-get install fonts-nanum* -qq # 나눔폰트 / 시스템관리
import matplotlib.font_manager as fm # 글꼴 관리
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  name='NanumGothic')
# ttf : truetype font
fm.fontManager.ttflist.insert(0, fe)
# rc : run control
plt.rcParams.update({'font.size': 14, 'font.family': 'NanumGothic'})

plt.rcParams['axes.unicode_minus'] = False

# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMYAAABXCAYAAABfoTq7AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAABqnSURBVHhe7Z1neFTFHoffsyWbXsgmkBAJLYQuEFQQEcQLCuIVFCmKNBVBUUSQpnQVERRElCsoiIgBBEFBAS/t0kEFJNIhPZCebHaTzbaz90MKyRKyC2xwE877POfDnpnZM2dmftP+M3MEq9VqRUJCohwy2xsSEhKSMCQkKkRY/3taje5KRdb2RKUUOJWcb+vkUvRpq2brqSzMluqRHe5KGf9qFsDWU1m2TjUC4fnlp6tHTtwiT7QKxNdDTvSxdFsnl+KroU0Z8/0FCk2irZNL4u+pYE6fBrz+/UVbpxqB1JWSkKgASRgSEhUgCcMOfgGeNFUrcLN1uAHuPu40ra3C19bhrkUgSO1JpL8coeSOTMBNBr4BnkSqlahsQlSMQK1AT5rWUtgttIJCSYM6noR5lL0p4CYX7IYtwVF/dwUh4YH0beFFYEkOItC6fT0md/SjVuk9ULkrCfK2vRR4ySC0aSiTHwuimfyaf2ehVClQX/fca5faU0a5x8rlBFbgr0K/TkfA11tJkK8nPbqEM/4+X+p4K1F7qujVsyn/6eZP26h6TOnkh7pM2oKMkLp+PHZvID0jvbmnRDWCggc6hDPpPm/KlvdrCHh5Fr1bcO1ajOxVj0EN3IrfV45v4zC+HBpOLwdrLGnwXYpA24cjGFtbwwcb07ggFt3r3D2SF90zmLw1i1Rr0b32XSJ4I8Km5hIL+ennOP5s0IDpzYwsXZPMUUtZD5Vjf/At0PzBRkxu7nbD2sycns6ULZlcLc5RZb1QFnX3x69cwSvCnJXBzM0ZxN9i7tsdfCv8eHNwXdoryt82p2ezSwzgkcKrrKIOw70yeefnTFKsgMyNro/UY0i4AqPeAu4K3At1fLc9mZ25Cnr2akx/81XG7MjlujlGmQcD+9and0AFL2vRsuooPNdBzsYN8WzNs/VwPfLW/35tpu3NmkST2p6olDL+TrkuKW0QCAkPpIO3gQNn88kqFkF4IzXtFAXsvKBHV+xTk5PP8cu57D1fdB3IVRBVV8bFs7mk1gqgS5CFP2LyijLbQf7dRs2vMVmYxRsHykjOZsv5ArINBv53KIkvDmew+bSOdKORY78n8/nJfLRlgsv9fHi8oZzjB5L4+q+c0vjuPZ/Lvrh8UvQiN6HdcrgrZXRrGsC2mGxbpyJEA0dPZXMiV8RkNHElQ8eeP1L4/Lie2o1r0cKi4y+8aetWwO7zBWgB9/A6jGur4sLhOGbtyeDXOAsNImvxkJeB3fFm6kfUooWo49fLhZhsn2c1c/pcJr9cKiDbIKLRGYi/msdvR1NYcjCXJH8/eofLuXxJS5IZTHZe/EaVz12IgJdKhsxNjncFlU5Z8vMKuZSuL71idSICYLlRZe8srCC6efBIOzUda4FoBVHlSbeoIO73t1KxpkQ0uQXl4nspXc+lbDMGW69OJrR5GFMeDqC+IGL09KZ/j3AGhxUVOWVYbQaGlU9oXx8lXlYDfycaMQImjZaYHPDwVuJrJ08A8PRjZO96PFtPgUlvweLjy/M96zOk+JnIPHi6TyRLeqmxefR1SMIoQelFc7WATOVJy9rlU00eFMTswZHMjnIv7ZcL3j483ak2XdUgU8hQWEX0xpIAKro+GMLQZh5O78dbTSKFCLgpiuIoKGQoyz67AgRBQCZw3WWnbNwmAuF1PHDLzmHF/lS+2ZPBSZOSxrWLuoIWjY4YTfkQOVmF5AgePNTKhxB3OSH1AukcBDnZheRUKPryKAK9aOpu4vDRFFYdS2Plngz+MitpGqoqKuhiIbv2JjD/YC7pdv5PEgYAAvc0D+R+dwvZBiWd2gRQt0zKiFodW35P45dEEyWNgszDgwci/WnpI+CuFJBZLRSYilJbEOQE+LlRx1PmxAQWaPVAA+b1DKS+INA0qgEf9WvEh90DCJXJafdgQz58LJBw2wcKKh7v3YxvX2xe/hoaRkeb/r9zsXI+XkderUDe7HUPE58M4X65gRNJBkRA1OZzumy/DzBdyeDrU3oCmocxf3Ak87vXIiAzk6//zKcS3ZdiTsvjiEbBw13rM+HRMCY+GUJ7Qc/huMLiLqNIRno+pzNMdltLaYwBBNSrw5sdvTFeSGb+31aiWgXS3stITLIRdUM1bdHw7eEczudfy0iZlxePRnqQF5/F32Z3QuVGTsTqEUMD6BJYyOofk1iXck1I9nBkjGE0mElJz+eP2DyOZ8to1dgL3cWrfPdXHn8m5XM6VU9SnqW0/201mkhK1XH0ch7nRXdaBRjZvjuVLZfyOByn41KuGb3NMxzF7hgD0GfncTDZSKHViiZbx5FLOtKVKvwEC1ey9KQUylAUFHD8irE4HlbSr2jYdUHHmStaDvydztq/dFw1A4KciMrGGAAWIzGXdWQrFLgZjSSk5LL5SAYHc60Ibkpqy8ycSyoo+j872NYvdx3eYXWY8mgA3qnpLDmWz9XEVBYe1KJsFMKIZtfPAMnkMtyVMjyVAgKgVCmRZ+aw9ng+ZrU7vrYBnEhupo7jidqiK9WIHit52cW/E7Ucv2IoN1tj1RdyqtjtXJ6I1WohKanEfyHZN9ag08jL1LL/ikh4ZDDDHw5h4L0BtA/x5L6oerzWTE5KSkFRN0kQUCmL0lZuNpKQqSdFL+Dr60ZYLXcaBShwqIFTePBQ6wAaywwk5Imo/D1pEepFyzAfOjTyIlRpG6BipOla5ETU9yAvSUdamZkKLx8lgtZM23LTtQLtuzZhbONrxqpyiAXsj5XTsZ7B6dO1DZqH0qduGSOZ0o0mIW4Ys3TElVGDqNWw6JiRHp2DaFnGKunu60lTP5GE5LL9dSvpsWl8d7nC+rdS7E7XliKnS48IhvtpWLTlKicLi28rVPTt1ZCnhAze+TmTVHUw7z+ppmScXA7RSn5qFvtlgTxqusF0bTGCZwCTB9QhwmxGV6ZlEBRy/JWFrHNwuraiaNxlWLgYX14UAPlaU+n07DWsnDmZxEfbE5i3LYG5v8QzZ0scM36KZerGS0xYn8y+gqqpZyyiiMF87SrUF3IqNo9zmvL3DWYrYMVU7p6IJlvH0bgCUk3l/RpvQry3hki61gQeHjwQ4UvrOh40ru1F+wh/WvhAvtaI1gqW7ByWb09g7q8JvL8ljpmbY5m84SJj15xjxMqzvLItp3gK3THSE7P5+WQmPxVfW+INNzU1LbUYlVKRga9yGt7XsIoMfK6F4y0GIFfStqWaR+t7EuYlRyVYyc83Epucw9ZTeSQ6MrIWlJUb+IoR3H0Y/GggDSqq8kUDu/Ze5eCNApdBEoYdVB5KfLCQrRcdGkgrVAr8lVa0OovdmY+y1GhhOAkPDyVeWMjSi1R1oa1IVxJlMOhNZDooCgCzwUzmTYpCwjH0xXlR1aJAEoaERMUIr605fycE+I/xr6YBeLvL2Xwy09bJpVjwbGPe2RSLwexo2/TP4uch5+3H6/HupjhbpxqBIB2fIyFxPVJXSkKiAgSD0VyjWwyZrMhCbalkqYUroFTIMFWTbhQUWaqVcqF6xfkmEE7FOrJusfqi9lOhkAmk5pSYXF2T5uF+nEvU3GDpuOuhkAs0DvXhXJIDZuRqiNSVkpCoAEkYEhIVcMvCEAsSOROTSEHN7GJK3OVULgzDL7w/aBbHKjDjiglreG/6GpIlYUjUQCoXhkVLRlIaBbb371LEvBi2zHuZMUMHMuXjzcRVUGG4FiL559fyycebbB2qBaLuDFvmvs+OtDK1r5jOn99MYNywfoybtpzjWVVTM1cuDDuIWb/z05efsvlYhq1TzUO8yrZ3R/Gzx2Amz59NN+1ipn58+JZ3wFU1liu7WDHhCfoPmcLGo4m2zq6NJZmD/3mV4U8+zby1B0gpLJmqsxC/+lXmn2zL6AWLGVpnKzOnryejCrRxW8JA6Udw3XtQ+zl6Tl/1Rcz5jW2nO/DiK10IC25C9wmvE354I3+5aKthzcvEEDmGhZ88T53by+U7j1VDjqE5Q5d+SZ+6ZY6TMJ9m66ZCerw5kCbqUNqMGEPUuY3srYJWw06SyRDyd/PZ808w4rk+jBzaj9FD+zJyyCR2ZIPMtwkdez/NQ5F+tgFrHoUFFMqV17ZXKoPxN8STkOf8THEGiqYDGP3yEzTwqHCvoWujaEHvsWPo1tCj/E7J/BguaiNoFlIsFlVLmtRN4nLcze9AtEflwvDsz8IT59i44RdWfL+ZZas2sHTVJpZ9O4/Hatl6rtnIanfmPu89bNgej5FCknas5VCaAZPz80TiBlg02Wg9A/ApUYvMFx9PPfk2p404g8qFAZB/jv2bvmfzhvLX1lNqHn++K2r7/1AzULRk+ILJBG8dzfB+z7LwXCPaNayFX1WefiBRHpkMwWIus0XVjFlUoHRzfqtoN1ctubv55qNoLuRo0OZdu/KSd7N+9V4yXbMnUSW4NezLW8u3sWbDFhaMrIPOoxmR7ra+JKoKea1QAgtSSSs55MCcwtWsEMLucej8kJvCrjAA8G5JtyGjeWHEteu5J9vi71joGoPFUlJXFXD+u2i0XfrQ0Pl5InEj3Dtwf0QM+w/mIgLm2O0ck3WmU5izz3t0VBgSgJFTiwfy0mujmfxiH+bGPsPbw5o6dtaRhHOQhfDkG0PI+uRpxo1/kVcm7ue+Ca8QUQWZYHd1rZjxIx++/iVnKzi9TdFwMLPnvsA9zhes03Dq6lqLlrTYeHRe9akf6uPUc2ml1bWOI+pTSYjX4B0eQZBn1dTtdoVR3XGqMKoQSRiuRdXITUKimiMJQ0KiAiRhSEhUgLTn20Wodnu+q2mcHUXI0Rpcu8TcJiqlDEEQKKz604tvCz8vJXn5pjtyyp4zkAng5aFEW1Az18QI6Tn66pIXt4SHSo5MEMgvrGC+2YVQ+6nI0hiqlTD8fVRk57no8uLbRBpjSEhUgCQMCYkKuA1hiORePEJMsiMfN5CQqF44IAwDKYe/Y8mH77Mkej9JpQZkE6eXv8J722/8cUIJieqKHWFYiFszlGcm7kJsEIFvzHsMfvErLrv2BI/TETUxRE+ZzqYrZV7ceIroaW8wcXzxNXEJhzQuNHXp6vGrFDOpBz5n8tDePN1nIFO+PkJ2SdTFNA4ufZ2hfXsy5M3FHEyvmneqXBiG/Xy5KJUBy5bzxoD+DJnzFSOM7/JY+4d5qndvpv2S5vAHVaollkT+++HzPNb538z4/iDJZSdgDKfYsU1D6/4vMPC5Fxg4sDuNvSpPzjuKq8evMgyHWL0mg4enfsuKxSPw3vASE39MR8TC5eUjmXHyAaauWMVbEbuYMO5bUqqgoq40pczxR4jx6EzXRsXreuWhPNLlXoJ7vMf6rVuZ80Ttyv+gumPVkifvwPi1Kxhks4TYkpON1q8xbaLuo13UfbRrF0lwFSx/vlVcPX6VonqItz6fSa/IWvjX+xcjn23E34dPYzLH8MM6PU9NGkRkQDBtXhpL54s/sP2q86vnSsu1qNWh9/bFt7RMyPHz88ZYUFBt5ttvC0ULnnn7dR5vbLMpH7BqcsmT6Uj6/Rhnruhu6ougdwJXj1/lyJCXlkwzKSnp+AcHI9ed4qyuCa1LTg5RtqR5eCIXYp1vZKxUGPIgNb5ZaaSX2sbMpKZm4ekBaUlXyCqoXsntVFQRPNxcz76Ny/nwuY50f+07LruSDdHV4+cQIrlHPmLGLw0ZNqgpQm4WGo9a+JYehuCNr5fxzh+GIK/blY6ee/j5qLZoLFF4ip92ZeGb8g0zp05mxeHsmj3GqARFk0FM+3Qxcxcs59udG3k2cS4Lt1//ZfB/ClePn11EDTGrRtJ/0jmeWvYf+teTFx2GIJY9DEHEbFagqIJjzSoVBorWjJz5OCfe+jdjZ0xj/IDRHHrkS9auWcvK1d/y9qNBdv7gLkFRn/ZtPLmSmGXr4hq4evyuQ8vvC/ozdkcrPtj8LSNa+yAD5IFhBOmucqWk52ROJiWzNuH1nD94slOuZQR0eZ+ftnxMn6gonnzvJ9ZPehBvW293IaLFfK21LDjD7iNmmraqU97TP4irx68yLAmrmPdLa95bPpb2AWWKqEdHOjc7yX/3FPVUzLG/sM/yCN1KJoecyG0sIjRwcOL9LG6yg3UvuW6CO2URofEIM7tNJ3D1Nl5vIAdE0lYPZeBaI83re5L1dwzmHvP4YsqjBNupam6EcxcROj9+tlTlIsLCna/SYdR+QhqpS2tuebOXWfrxcwTGfMbwV6NR3NsY/blM2s9ew8TOAfZq+JtGEsYtYyQn8QKJ2RBQvwn1/G+vo+tcYeD0+NlSlcKwS0EqF2Nz8WnQhDpVZJu5DWFAYXYKeco6BPs487wM51J1wnAuzhdG1fKPCuMOcFtyc69V16VFISFxq9yWMCQkaiqSMCQkKkDI1dXsPd9uChkyQaDQ5NpWel/Pov3T1SUzZICnhxKd3vnLMVyBGt9iCILtKicJpyAUnb5SU5GO6HQRpCM6XYsa32JISNwKkjAkJCrAcWGIlutW0ppS/+ZUbNFHPOxjJO3sH8Q7sr3SHMeB9TuIL2uTEzO5fPwsWa49hpaoIdgVhiVlG4tGdOWZvr0Y/lQ3Rs78ofjD7yK5v03nnW9OYWtT1qfG8MfenfxxNo1Su6iYze65o1hz+kazGGa06YkkJyWQnLCPDYtWciQhgeSkBFLSNZgNh1k+5n2O1ExDq4SLUbkwLIlsmDaLhJ4riP5pB6s2rWOw6WMGd3uIwf168saKk+V3hom5HP/iWV4YPZ/fju5j2ycDeeHVLzlbUNbTDRBzOb5mNovmzWDRwj0o2rnzx8IZLJo3g8XfHULzjwxKRTJ//4r5rw9g5IjhzF/7J9c3eCKZ+xazYM3v1yoBV8GYyIFl43lrSB9eGTWPIy766eXrEHM4/eN7TB/Vn1deHs2SrecoLUJiOn9+M4Fxw/oxbtpyjlfBN76xKwzjn/xxsRW9ejfEDUAWRIenHiG40RA+Wb+NxSPalPuqkPnsUj7a3oJpa75l6qTZTPtyM+PUq5m/5pL9rZUyNV3GLWPqi10JtmrIzsgkp8CDBr2mMOOtngQKABb0udnk6e3+m3MwHOKH6BTuHbOYudMHIKwbyQe/ZpbvOuqP8M2cOUTvOY9LnbAlZrN/9mCWJD7IqE9WMmdsL8KUlWe3yyCmkKhtwYDpy/hg4qOkLR7G58cMgIX41a8y/2RbRi9YzNA6W5k5fT0ZVaCNylNKHkaITywX40uy3ELmhfNkXd3HqgVz+Hp3QrlCYoq7QE6jdjTxLL4h86N163CuXr58XXerIsT0tcyesAX1S8tZtm4rX8zug+7LoSw6WDzVajrJD1NeY8n2ZNugVYOqIyPnz6BHZG0C6/dgSJ96nDl6usy7GDi3YgEXGj1IkItN6psvrWT58U68Pe0ZmqgDCG7WijAPW18uiqIlPYf2pUWoP4ERvXksykBcbD6i+TRbNxXS482BNFGH0mbEGKLObWRvFbQalQvDLYrB46I48OazzP50IctmD+GtH8N4dcpLdH7wYdo18i9n5HFr9QDhZ7awJ6VYSIZYtv12kSZtW6Es4+9GiJorZCoa0zJSjQJQhbalWYiezPTiTooyiiFLo5naN9w2aBUhR17aJJpJS8vEN+jarkXDX4v5+EhHXusXbich7zQW0g/sJbPNwwTHbGbT+s2cSHFtO86NsKTvZ9+lRnS6zxdZfgwXtRE0CynOFFVLmtRN4nLcjcatt46d/JQR3P0jvlk1ky4NQgh7cDyLohczsFtXOjzUhfvv70jL+r6l4pCHv8T0cUFsfrk7I4b3Y/jTw9jZeDZT+oXaexAAikbDeKN3Mp8M6M2YUYMZ1f9ZNvhOYFQvP1uvdxiRvD8XsnB3BAOfiUQBiJr9fDb3GJ3ffYOmjqj+jmImJfEKnFnJwi2XyLvyK/MGDyb6kiPttmtg2P8eI5/pxGM9xpPSYxJPNlBg0WSj9QzAp/QwBF98PPVVchiCQ5Zv48G5vPnFIa5fVSVD3WMO7w1rbdMiGNBm5iHzDcKrZH+MmEr0sN7EjjrIOw+qyvkuh6gj7VIiRr9A/PyD8FUVS8p8ib3RZ7ln0JPczE7G27Z8ixrOrp/MnDWF9Jn3Gf2aeyMzx7F57IsceGQl8/qFY/7fG/T97n7WLR+Mj214B3Gu5dvAwalRTM56h81LBxEoM3Ph08eYWjiPDZPa23q+Je6U5duU+Scbpo9lZ9tlfNFzD6+OTOONzbO51w1Aw+aR3Tj1wiGmd66kTN0CjlTkuHUYz8dfrWPJ12WvtczqLnAmNreChW8qfNRlRHEzmI7z1eh3OSCUEQWA5QI7l63n3B0d4Wo5ueR5pu1uyaTVX9O/uTcywJL4P46mK8nc8BovDezNqHk7yTixmLFTom3/4B9CTqBajW9wKD4yAAVhYSEU5FS/c4aV6iieGdKJjP0H0PiHEliQSlpJw2dO4WpWCGH33ERN6SAOtRiG38Yx8OMTePmoyo0prAVZWLt9xjcTOiLP+y9LJ67gnAVE0YTFaMZkNiNarWAVqN1rEq12jSPeXoth2Mf7PYZxMCCCwLLva9WQdKkxbx/+jidKBvcOcDsthiVpKaNfS+KVtR8QVckzDS7XYoDx6GT6zVIxe8Ms2rhr2D+tF9HN1/HF82G2Xm+JqmwxRE06Oe7BBKoAzCSufJZxsWP5flYkW155mr/6b2NWd3/EC5/w8jsmJkdPItLJ2nBIGAWbX+DpA4P4aUEvblykC8nL0GJSKFDKFSiUbripVChKKn1Hu1KGfbzfaxENozcwqOzOfcOvvPuvaDruWH3HhGH431h6jz9I7fqBpRWCotnLfDSrH0Flo+aCwsByhV0zBvFJTBgtgtNIlA9g1qcvE1lJ0t8MVSkM46HJDJp2grDWjfHWXeSyqQtj5k/hoSAZhpiljJv4A8rmDdBdziZqykpGPeDvWNfnJnBIGIZ9s3l18cEK5+kF/+6M+3wCbe0luKPCMJ5g1dip7KxoctqtE6NWTqdTJcFtuR1h3EmcLgwAzOQlXyDVHET9+kFFtignUZXCADBkxZGQqkHu34Dwun6UbRBEfSoJ8Rq8wyMI8nS2JIpwSBjOwUJ+Rhom31D8b6Jg3y53tzCqjqoWxj9N1citQuR4Bd1ZUUhI3Cp3UBgSEtUHSRgSEhXwfxBCdKlgGpIXAAAAAElFTkSuQmCC)

import pandas as pd
data = pd.DataFrame({
    "사과": [9, 15],
    "포도": [1, 5]
}, index=["남", "여"])
print(data)
total = data.values.sum()
gender_total = data.sum(axis=1) # 방향이 열방향
gender_total # 주변합

# 남자일 확률
p_gender = gender_total / total
p_gender # 주변확률

# 남자 중에 사과를 좋아할 확률 (조건부 확률) 과 여자 중에 사과를 좋아할 확률
p_apple_given_male = data["사과"] / gender_total # 선호과일 / 주변합
p_apple_given_male

# 위를 전확률 정리로 구하면
# 전확률 정리 = 조건부확률의 결합확률로 표현
(p_apple_given_male * p_gender).sum()
# 성별이 주어졌을 때의 사과 선호
# 조건부확률과 p_gender를 이용해서 합계

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
data = df[["sex", "survived"]].dropna()
print(data.head())

# 위 데이터의 교차표를 생성하시오
# groupby, crosstab, pivot, pivot_table
cross = pd.crosstab(data["sex"], data["survived"])
cross

# 성별 주변확률
total = cross.values.sum()
cross.sum(axis=1) / cross.values.sum()

# 생존여부 주변확률
cross.sum(axis=0) / total

# 결합확률
cross / total

# 결합확률
joint_prob = cross / total
joint_prob
# 여성이면서 살아있을 확률과 죽었을 확률
# 남성이면서 살아있을 확률과 죽었을 확률

# 남자이면서 살아있을 확률
joint_prob.iloc[1, 1]

# 조건부확률표
# 성별로 살아있을 확률
# 행의 합계나 열의 합계로 정규화
# 인덱스는 행에다 줌
# column은 열
print("조건부확률 P(survived | sex)")
cond_survived_given_sex = pd.crosstab(
    data["sex"],
    data["survived"],
    normalize="index"
)
print(cond_survived_given_sex)

# 생존여부별로 성별확률
# 정규화를 이용한 조건부확률
# P(sex | survived) given survived sex probability
pd.crosstab(
    data["sex"],
    data["survived"],
    normalize="columns"
)
# 조건부확률 : 열기준으로 정규화

from scipy.stats import chi2_contingency
chi2, p_value, dof, expected = chi2_contingency(cross)

# 성별이 주어졌을 때 생존여부
cond_survived_given_sex

# 생존했다고 할 때 여성이었을 확률
cond_survived_given_sex.loc["female", 1]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from scipy.stats import norm
# 유방암 데이터 / 양성 : 악성
data = load_breast_cancer()
df = pd.DataFrame(
    data.data,
    columns = data.feature_names
)
x = df["mean radius"] # 평균 반지름
print(x.head())

df.head()

plt.figure(figsize=(8, 4))
# 범위를 안주면 카운트가 안됨
plt.hist(x, bins=30, density=True)
plt.title("Mean Radius Histogram")
plt.show()

# 데이터의 평균과 분산을 구함
mu = x.mean()
sigma = x.std()

print("Mean =", mu)
print("Std =", sigma)

# 분포 : 정규분포를 가정하고 밀도를 구함
pdf_values = norm.pdf( # 밀도를 구하는 함수
    x,
    loc=mu,
    scale=sigma
)

print(pdf_values[:5])

# 우도를 구함
# 각 데이터의 밀도의 곱으로 계산
likelihood = np.prod(pdf_values)
print(likelihood)

# 우도를 구하기 위해서 작은 수를 계속 곱하면 0이 됨
# 그래서 log0likelihood를 사용함
log_likelihood = np.sum(
    np.log(pdf_values)
)
print(log_likelihood)

# !pip install distfit

from distfit import distfit
dist = distfit()
dist.fit_transform(x)

print(dist.model)

best_dist = dist.model['name']

params = dist.model['params']
print(params)

from scipy import stats
distribution = getattr( # 분포
    stats,
    best_dist
)

pdf_values = distribution.pdf( # 밀도를 구하고
    x,
    *params
)

# 로그는 덧셈이 곱셈
# 곱셈을 덧셈으로 만들기 위해 그냥 log를 취하기도 함
np.sum( # 밀도를 곱해서 우도를 구함
    np.log(pdf_values)
)

# logistic regression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

# Bunch -> DataFrame으로 변환
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_original = pd.Series(data.target)

# 정수형으로 범주화
y = (y_original == 0).astype(int)
print("데이터 크기:", X.shape)
print("클래스 분포")
print(y.value_counts().rename({0: "benign", 1: "malignant"}))
# 양성과 악성 (malignant)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.25,
    random_state = 42,
    stratify = y
)

# logit link 함수 = linear regression
# log odds
# sigmoid 로 변환해서 실제확률과 비교 : 승수비
# 우도를 구하고 MLE를 이용해서 우도가 최고가 되는 분포를 구하고
# 이 값을 이용해서 solver를 이용해 선형회귀의 계수를 조정
# solver는 방향과 곡률을 결정하면 Newton
# 방향 : 경사하강법, qunsi-newton(곡률을 간단하게)
# solver가 선형회귀의 계수를 수정
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000)) # 최적화 default 100
])
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# 0.5 기준을 변화시키면서 recall / precision 을 조절해 최적으로
def predict_by_threshold(y_prob, threshold):
  return (y_prob >= threshold).astype(int)

def evaluate_threshold(y_true, y_prob, threshold):
  y_pred = predict_by_threshold(y_prob, threshold)
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

  return {
      "threshold": threshold,
      "accuracy": accuracy_score(y_true, y_pred),
      "precision": precision_score(y_true, y_pred, zero_division=0),
      "recall": recall_score(y_true, y_pred, zero_division=0),
      "f1": f1_score(y_true, y_pred, zero_division=0),
      "TN": tn,
      "FP": fp,
      "FN": fn,
      "TP": tp
  }

# 디폴트
threshold_default = 0.5

# ROC 기반
# falese positive rate / true positive rate
# positive로 예측했을 때 true/false 비율
# tpr = 민감도/재현율로 나눈 값 -> 높으면 양호
# 재현율은 실제 양성을 얼마나 잘 맞추었는가
# 민감도는 실제 음성을 얼마나 잘 맞추었는가
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
youden_j = tpr - fpr # 오진
best_roc_idx = np.argmax(youden_j)
threshold_roc = roc_thresholds[best_roc_idx] # 가장 클 때의 경계값

# positive를 중심으로 판단
# precision : 정밀도 : positive로 예측했는데 얼마나 잘맞추었는가
# recall : 재현율 : 실제 positive를 얼마나 잘 맞추었는가
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
# f1_score는 precision 과 recall이 중요하니까 이 둘을 조화평균한 내용
# accuracy한 값만 보는 것보다는 정밀도와 재현율을 동시에 고려함으로 모델 일반화된 평가를 더 강조한다
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
best_pr_idx = np.argmax(f1_scores)
threshold_pr = pr_thresholds[best_pr_idx]

# 비용기반 임계점 분석
# f1-score, accuracy를 기준 분석 오탐, 미탐을 동등한 가치로 봄
# 0.5 0.48은 정상이라고 판단
# 비용기반으로 정수화해서 : 잘못 판단한 경우를 고비용
FP_COST = 1
FN_COST = 10 # 미탐을 중시하는 점수를 부여
threshold_grid = np.linspace(0.01, 0.99, 99)
costs = []
# fp : positive로 예측해서 틀린 것
# fn : negative로 예측해서 틀린 것
for t in threshold_grid:
  y_pred = predict_by_threshold(y_prob, t)
  th, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  total_cost = FP_COST * fp + FN_COST * fn
  costs.append(total_cost)
costs = np.array(costs)
best_cost_idx = np.argmin(costs) # 비용이 가장 작을 때를 기준
threshold_cost = threshold_grid[best_cost_idx]

# 확률로 판단할 때 임계점 분석은 모델의 성능을 높여줌
threshold_methods = {
    "Default 0.5" : threshold_default,
    "ROC Youden" : threshold_roc,
    "PR F1 Max" : threshold_pr,
    "Cost Min" : threshold_cost
}
results = []
for name, threshold in threshold_methods.items():
  result = evaluate_threshold(y_test, y_prob, threshold)
  result["method"] = name
  results.append(result)
result_df = pd.DataFrame(results)
result_df = result_df[
    ["method", "threshold", "accuracy", "precision", "recall", "f1", "TN", "FP", "FN", "TP"]
]
print(result_df.round(4))
# FN : negative로 예측하고 틀림
# 오탐과 미탐
# Cost Min : 암환자를 못찾는 경우가 0건
# FP : 환자가 아닌 사람을 환자로 예측 : 과도하긴 하지만 미탐보다는 낫음

# ROC 커브 시각화
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC Curve AUC={roc_auc_score(y_test, y_prob):.3f}")
plt.scatter(fpr[best_roc_idx], tpr[best_roc_idx], s=100, label=f"Best ROC threshold={threshold_roc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.title("ROC Curve - Youden Threshold")
plt.legend()
plt.grid(True)
plt.show()
# false positive : 양성으로 예측했는데 틀린 것
# 양성이 아닌 정상을 양성으로 오진 - 기준값은 없어야 함
# true positive rate : 암을 암으로 판단 -> 높으면 높을 수록 좋음
# Balance point : 균형점

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"PR Curve AP={
    average_precision_score(y_test, y_prob):.3f
}")
plt.scatter(
    recall[best_pr_idx],
    precision[best_pr_idx],
    s=100,
    label=f"Best F1 threshold={threshold_pr:.3f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - F1 Threshold")
plt.legend()
plt.grid(True)
plt.show()
# recall : 재현율 : 진짜 positive를 얼마나 잘 잡아냈는가
# precision : 정밀도 : positive 예측해서 얼마나 맞추었나
#                   real positive                 real negative
# positive 예측     TP(양성)                      FP(오탐 : 환자가 아닌데 환자)
# negative 예측     FN(미탐 : 암인데 정상)        TN(음성)
# recall = TP / (TP + FN)
# precision = TP / (TP + FP)
# sensetivity(민감도) = TN / (FP + TN)
# 평가값에서 precision 과 recall 을 고려한 평가가 F1-score
# 조화평균
# precision vs recall 은 trade off 관계

# FN(미탐), FP(오탐)을 기준으로 최적의 threshold를 결정
plt.figure(figsize=(7, 5))
plt.plot(threshold_grid, costs)
plt.scatter(threshold_cost, costs[best_cost_idx], s=100,
            label=f"Min Cost threshold={threshold_cost:.3f}")

plt.xlabel("Threshold")
plt.ylabel("Total Cost")
plt.title(f"Cost-based Threshold Optimization FP={FP_COST}, FN={FN_COST}")
plt.legend()
plt.grid(True)
plt.show()

#multi - class classification

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names
print("클래스:", target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y # 층화 추출
)

ovr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        # 0,1,2
        multi_class="ovr", # one vs rest 3에 걸쳐 적용
        solver="liblinear", # 2개 분류 - 3개가 안됨
        max_iter=1000 # 가장 강력한 SVM (support vector machine)
    ))
])
ovr_model.fit(X_train, y_train)
ovr_pred = ovr_model.predict(X_test)
ovr_prob = ovr_model.predict_proba(X_test) # 확률예측
print("Accuracy:", accuracy_score(y_test, ovr_pred)) # Accuracy: 0.8

softmax_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        # 0,1,2
        multi_class="multinomial",
        solver="lbfgs", # 최적화 (qunsi-newton 곡률을 간소하게)
        max_iter=1000
    ))
])
softmax_model.fit(X_train, y_train)
softmax_pred = softmax_model.predict(X_test)
softmax_prob = softmax_model.predict_proba(X_test) # 확률예측
print("Accuracy:", accuracy_score(y_test, softmax_pred)) # Accuracy: 0.8

print(confusion_matrix(y_test, softmax_pred))

from sklearn.metrics import classification_report
print(classification_report( # 분류
    y_test, softmax_pred, target_names=target_names
))
# accuracy : 개별요소들로 판단
# macro avg : class별로 해서 평균을 낸값
# weighted avg : 참여한 샘플 수에 가중치를 부여해서 평균을 낸 값

compare_df = pd.DataFrame({
    # 실제 데이터
    "actual": [target_names[i] for i in y_test],
    # 예측 데이터
    "OvR_pred": [target_names[i] for i in ovr_pred],
    "Softmax_pred": [target_names[i] for i in softmax_pred],
    # 확률 예측 : np.max 최고 확률을 계산
    "OvR_max_prob": np.max(ovr_prob, axis=1),
    "Softmax_max_prob": np.max(softmax_prob, axis=1)
})
compare_df

# 문제
# 다음 데이터에 대해서 logistic regression을 실시하시오
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, GridSearchCV

# PCA가 시각화 사용됨
# from sklearn.decomposition import PCA
# PCA : 차원축소
# 원래 original data의 변수 간 공분산 / 상관계수행렬 (대칭행렬, 정방행렬)
# 고유값분해(eigen 분해)
# 고유치(분산크기에 따라서 정렬) + 고유벡터
# 고유치를 확률해서 합하면 1.0 -> 0.85까지의 변수만 선택해서 분석
# 주성분
# 주성분 중에 제1주성분, 제2주성분 : 제일 분산이 큰 것
# 2차원 그래픽으로 출력 가능
# 변수가 100개인데 2개만 선택하면 98개는 고려하지 않는 것 아닌가
# 모든 변수가 다 참여해서 새로운 축을 만듬
# 그래서 모든 데이터가 다 참여한 것과 같은 효과가 있음
# 2개만 선택해서 출력해도 의미가 있음
# 100개의 변수에서 2개의 변수를 선택하는 것과
# PCA 이후에 주성분 2개의 변수를 선택하는 것과의 차이는
# 하나의 주성분은 모든 변수가 참여해서 결정

wine = load_wine()
X_w, y_w = wine.data, wine.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X_w, y_w, test_size=0.3,
    random_state=42, stratify=y_w
)

wine.target_names

X_w.shape # (178, 13)

pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        multi_class='multinomial',
        solver='newton-cholesky', # 완전한 뉴톤법
        C=1.0, # 규제강도 - 역의 값
        max_iter=1000,
        random_state=42
    ))
])

param_grid = {'lr__C': [0.001, 0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(pipe_lr, param_grid, cv=5, scoring='accuracy', n_jobs=1)
gs.fit(X_tr, y_tr)

best_lr = gs.best_estimator_
y_proba_lr = best_lr.predict_proba(X_te)
y_pred_lr = best_lr.predict(X_te)

print(f'\n  최적 C = {gs.best_params_["lr__C"]}')
# 테스트 정확도 : 0.9815
print(f'  테스트 정확도: {best_lr.score(X_te, y_te):.4f}')
print(f'  5-Fold CV: {gs.best_score_:.4f}')
# 변수가 13개라서 시각화 불가능
# PCA(principle component analysis), MDS(multi dimmension scale)

COLORS = {
    'primary':  '#1E88E5',
    'success':  '#00C8AA',
    'warning':  '#FF8F00',
    'danger':   '#E53935',
    'purple':   '#7B1FA2',
    'green':    '#2E7D32',
    'dark':     '#1A2340',
    'gray':     '#546E8A',
}

from sklearn.decomposition import PCA
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('LogisticRegression - 와인 품종 분류', fontsize=14, fontweight='bold')
plt.close(fig)

ax = axes[0] # 첫 번째 출력 축
# 13개 변수 -> 2개의 변수
pca = PCA(n_components=2) # 2차원으로 차원 축소 (제1, 제2)
scaler_tmp = StandardScaler()
# 178 x 13
X_sc = scaler_tmp.fit_transform(X_w) # vector 중 거리기준일 때
X_2d = pca.fit_transform(X_sc) # 178 x 2
lr_2d = LogisticRegression(multi_class='multinomial', C=1.0, max_iter=1000, random_state=42)
lr_2d.fit(X_2d, y_w) # 분류모델

h = 0.05 # x축 간격
# 그래픽여백을 주면서 좌우 상하 그래픽 사이즈 결정
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
# 사각형내에서의 정점좌표를 생성
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 정점좌표자체를 예측하면 자동 영역이 생성 - 배경
# 예측할 때는 1차원 -> 시각화해야 하니까 2차원으로 전환
Z = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# 등고선 -> 높이값에 따라서 0,1,2 -> 3가지 컬러값 사용
cmap_bg = plt.cm.get_cmap('Pastel1', 3)
ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
scatter_colors = [COLORS['danger'], COLORS['success'], COLORS['primary']]
for i, (name, c) in enumerate(zip(wine.target_names, scatter_colors)):
  mask = y_w == i # 실제값과 클래스를 비교
  ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=c, label=name, edgecolors='white',
             linewidths=0.5, s=60, alpha=0.85)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_title('(A) PCA 20 결정 경계')
ax.legend(fontsize=9)

ax = axes[1]
C_vals = [0.001, 0.01, 0.1, 1, 10, 100] # 규제강도
# 최적의 파라미터
# 최적의 파라미터별 결과값 : mean_test_score, std_test_score
cv_means = [gs.cv_results_['mean_test_score'][i] for i in range(len(C_vals))]
cv_stds = [gs.cv_results_['std_test_score'][i] for i in range(len(C_vals))]
# x축의 거리값을 균등하게 하기 위해서 log
ax.semilogx(C_vals, cv_means, 'o-', color=COLORS['primary'],
            linewidth=2, markersize=8)
# 표준편차의 신뢰구간 (x축은 한개 y축은 범위값으로 2개)
# 하한값과 상한값의 구간값을 출력
ax.fill_between(C_vals,
                [m-s for m, s in zip(cv_means, cv_stds)],
                [m+s for m, s in zip(cv_means, cv_stds)],
                alpha=0.2, color=COLORS['primary'])
# vertical 수직선
ax.axvline(gs.best_params_['lr__C'], color=COLORS['danger'], linestyle="--",
           label=f'최적 C={gs.best_params_["lr__C"]}')
ax.set_xlabel('정규화 강도 C (log scale)')
ax.set_ylabel('CV 정확도')
ax.set_title('(B) C 값에 따른 정확도')
ax.legend()

# tree ensemble 변수 중요 (variable importance)
# 회귀에서는 변수가 기울기 크면 중요변수
# SHAP XAI
# 계수를 이용한 변수 중요도
# 계수가 크면 중요한 변수
ax = axes[2]
coef = best_lr.named_steps['lr'].coef_ # 계수 : 클래스가 3개 열이 13개
# 3 x 13 하나의 클래스마다 선형회귀식이 만들어짐
importance = np.abs(coef).mean(axis=0) # 절대값 평균 - 행 방향 - 열 평균
# 역순정렬 / 그 중에서 중요한 10개만 선택
sorted_idx = np.argsort(importance)[::-1][:10]
# horizontal 수평으로 barplot
ax.barh(range(10), importance[sorted_idx][::-1], color=COLORS['success'], alpha=0.85)
ax.set_yticks(range(10))
ax.set_yticklabels([wine.feature_names[i] for i in sorted_idx][::-1], fontsize=9)
ax.set_xlabel('|계수| 평균')
ax.set_title('(C) 특성 중요도 Top 10')

plt.tight_layout()
fig